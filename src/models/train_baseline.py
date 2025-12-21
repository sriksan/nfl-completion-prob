import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent          # src/models/
src_dir = current_dir.parent               # src/
project_root = src_dir.parent              # project root/

sys.path.append(str(src_dir / 'data'))
sys.path.append(str(src_dir / 'features'))

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import pickle
from datetime import datetime

from merge_sources import create_modeling_dataset
from contextual_features import add_contextual_features

print("="*70)
print("NFL PASS COMPLETION PREDICTION - BASELINE MODEL")
print("="*70)

# Step 1: Load and prepare data
print("\n1. Loading data and adding features...")
df = create_modeling_dataset()
df = add_contextual_features(df)

print(f"\nDataset: {len(df)} rows, {len(df.columns)} columns")
print(f"Unique plays: {df['playId'].nunique()}")

# Step 2: Prepare modeling dataset
print("\n2. Preparing modeling dataset...")

# Features to use (only contextual for baseline)
feature_cols = [
    'down', 'yardsToGo', 'quarter',
    'is_first_down', 'is_second_down', 'is_third_down', 'is_fourth_down',
    'is_short_yardage', 'is_medium_yardage', 'is_long_yardage',
    'is_third_and_long'
]

# Check which features exist
available_features = [f for f in feature_cols if f in df.columns]
print(f"\nUsing {len(available_features)} features:")
for f in available_features:
    print(f"  - {f}")

# One row per play (currently we have 22 rows per play - one per player)
# We'll aggregate to play level for now
print("\n3. Aggregating to play level...")
play_df = df.groupby('playId').agg({
    'gameId': 'first',
    'complete': 'first',
    **{feat: 'first' for feat in available_features}
}).reset_index()

print(f"Play-level dataset: {len(play_df)} rows")
print(f"Target distribution:")
print(play_df['complete'].value_counts())
print(f"Completion rate: {play_df['complete'].mean():.1%}")

# Prepare X and y
X = play_df[available_features].values
y = play_df['complete'].values
groups = play_df['gameId'].values  # For GroupKFold

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Unique games: {len(np.unique(groups))}")

# Step 3: Cross-validation setup
print("\n" + "="*70)
print("4. CROSS-VALIDATION SETUP")
print("="*70)

# Group K-Fold: Keep all plays from same game together
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

print(f"Using {n_splits}-Fold Group Cross-Validation by gameId")
print("This prevents data leakage between train and test!")

# Step 4: Train Baseline Logistic Regression
print("\n" + "="*70)
print("5. BASELINE MODEL: LOGISTIC REGRESSION")
print("="*70)

lr_scores = {'log_loss': [], 'auc': [], 'accuracy': []}

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # Predict probabilities
    y_pred_proba = lr_model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    ll = log_loss(y_val, y_pred_proba)
    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred)
    
    lr_scores['log_loss'].append(ll)
    lr_scores['auc'].append(auc)
    lr_scores['accuracy'].append(acc)
    
    print(f"Fold {fold}: Log Loss={ll:.4f}, AUC={auc:.4f}, Accuracy={acc:.4f}")

print(f"\n{'='*70}")
print("LOGISTIC REGRESSION - AVERAGE RESULTS:")
print(f"{'='*70}")
print(f"Log Loss:  {np.mean(lr_scores['log_loss']):.4f} ± {np.std(lr_scores['log_loss']):.4f}")
print(f"AUC:       {np.mean(lr_scores['auc']):.4f} ± {np.std(lr_scores['auc']):.4f}")
print(f"Accuracy:  {np.mean(lr_scores['accuracy']):.4f} ± {np.std(lr_scores['accuracy']):.4f}")

# Step 5: Train XGBoost
print("\n" + "="*70)
print("6. PRIMARY MODEL: XGBOOST")
print("="*70)

xgb_scores = {'log_loss': [], 'auc': [], 'accuracy': []}

# XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'random_state': 42
}

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train model
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(X_train, y_train, verbose=False)
    
    # Predict probabilities
    y_pred_proba = xgb_model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    ll = log_loss(y_val, y_pred_proba)
    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred)
    
    xgb_scores['log_loss'].append(ll)
    xgb_scores['auc'].append(auc)
    xgb_scores['accuracy'].append(acc)
    
    print(f"Fold {fold}: Log Loss={ll:.4f}, AUC={auc:.4f}, Accuracy={acc:.4f}")

print(f"\n{'='*70}")
print("XGBOOST - AVERAGE RESULTS:")
print(f"{'='*70}")
print(f"Log Loss:  {np.mean(xgb_scores['log_loss']):.4f} ± {np.std(xgb_scores['log_loss']):.4f}")
print(f"AUC:       {np.mean(xgb_scores['auc']):.4f} ± {np.std(xgb_scores['auc']):.4f}")
print(f"Accuracy:  {np.mean(xgb_scores['accuracy']):.4f} ± {np.std(xgb_scores['accuracy']):.4f}")

# Step 6: Compare models
print("\n" + "="*70)
print("7. MODEL COMPARISON")
print("="*70)

print("\n                   Logistic Regression    XGBoost         Improvement")
print("-" * 70)

lr_ll = np.mean(lr_scores['log_loss'])
xgb_ll = np.mean(xgb_scores['log_loss'])
ll_improvement = ((lr_ll - xgb_ll) / lr_ll) * 100

lr_auc = np.mean(lr_scores['auc'])
xgb_auc = np.mean(xgb_scores['auc'])
auc_improvement = ((xgb_auc - lr_auc) / lr_auc) * 100

lr_acc = np.mean(lr_scores['accuracy'])
xgb_acc = np.mean(xgb_scores['accuracy'])
acc_improvement = ((xgb_acc - lr_acc) / lr_acc) * 100

print(f"Log Loss:     {lr_ll:.4f}              {xgb_ll:.4f}          {ll_improvement:+.1f}%")
print(f"AUC:          {lr_auc:.4f}              {xgb_auc:.4f}          {auc_improvement:+.1f}%")
print(f"Accuracy:     {lr_acc:.4f}              {xgb_acc:.4f}          {acc_improvement:+.1f}%")

# Step 7: Feature importance (train final model on all data)
print("\n" + "="*70)
print("8. FEATURE IMPORTANCE")
print("="*70)

final_model = xgb.XGBClassifier(**params)
final_model.fit(X, y, verbose=False)

importance_df = pd.DataFrame({
    'feature': available_features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop features for predicting pass completion:")
for idx, row in importance_df.iterrows():
    print(f"  {row['feature']:25s} {row['importance']:.4f}")

# Step 8: Sample predictions
print("\n" + "="*70)
print("9. SAMPLE PREDICTIONS")
print("="*70)

# Get predictions for a few plays
sample_plays = play_df.sample(min(5, len(play_df)), random_state=42)
sample_X = sample_plays[available_features].values
sample_probs = final_model.predict_proba(sample_X)[:, 1]

print("\nPredictions for sample plays:")
print("-" * 70)
for idx, (_, play) in enumerate(sample_plays.iterrows()):
    actual = "✓ Complete" if play['complete'] == 1 else "✗ Incomplete"
    prob = sample_probs[idx]
    print(f"Play {play['playId']}: Down={int(play['down'])}, YardsToGo={int(play['yardsToGo'])}")
    print(f"  Predicted: {prob:.1%} completion probability")
    print(f"  Actual:    {actual}")
    print()

# Step 9: Save model and results
print("="*70)
print("10. SAVING MODEL AND RESULTS")
print("="*70)

# Create models directory if it doesn't exist
models_dir = project_root / 'models'
results_dir = project_root / 'results'
models_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

# Save the trained model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = models_dir / f'baseline_xgboost_{timestamp}.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"✓ Model saved to: {model_path}")

# Save feature names
features_path = models_dir / f'baseline_features_{timestamp}.txt'
with open(features_path, 'w') as f:
    f.write('\n'.join(available_features))
print(f"✓ Features saved to: {features_path}")

# Save results
results_df = pd.DataFrame({
    'model': ['Logistic Regression', 'XGBoost'],
    'log_loss': [lr_ll, xgb_ll],
    'auc': [lr_auc, xgb_auc],
    'accuracy': [lr_acc, xgb_acc]
})
results_path = results_dir / f'baseline_metrics_{timestamp}.csv'
results_df.to_csv(results_path, index=False)
print(f"✓ Results saved to: {results_path}")

# Save feature importance
importance_path = results_dir / f'baseline_feature_importance_{timestamp}.csv'
importance_df.to_csv(importance_path, index=False)
print(f"✓ Feature importance saved to: {importance_path}")

# Summary
print("\n" + "="*70)
print("✅ BASELINE MODEL COMPLETE!")
print("="*70)
print(f"\nYour baseline XGBoost model achieves:")
print(f"  • Log Loss: {np.mean(xgb_scores['log_loss']):.4f}")
print(f"  • AUC:      {np.mean(xgb_scores['auc']):.4f}")
print(f"  • Accuracy: {np.mean(xgb_scores['accuracy']):.1%}")
print(f"\nModel files saved to: {models_dir}")
print(f"Results saved to: {results_dir}")
print(f"\nNext step: Add spatial features (receiver separation, air yards, etc.)")
print(f"Expected improvement: 10-20% reduction in Log Loss")
print("="*70)