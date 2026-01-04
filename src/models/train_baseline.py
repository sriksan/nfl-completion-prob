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
from sklearn.model_selection import train_test_split
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

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

# Step 3: Train/Test Split
print("\n" + "="*70)
print("4. TRAIN/TEST SPLIT")
print("="*70)

from sklearn.model_selection import train_test_split

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {len(X_train)} plays ({y_train.mean():.1%} completion rate)")
print(f"Test set:  {len(X_test)} plays ({y_test.mean():.1%} completion rate)")

# Step 4: Train Baseline Logistic Regression
print("\n" + "="*70)
print("5. BASELINE MODEL: LOGISTIC REGRESSION")
print("="*70)

# Train model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

# Predict on test set
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
y_pred_lr = (y_pred_proba_lr >= 0.5).astype(int)

# Calculate metrics
lr_ll = log_loss(y_test, y_pred_proba_lr)
lr_auc = roc_auc_score(y_test, y_pred_proba_lr)
lr_acc = accuracy_score(y_test, y_pred_lr)

print(f"Test Set Results:")
print(f"  Log Loss:  {lr_ll:.4f}")
print(f"  AUC:       {lr_auc:.4f}")
print(f"  Accuracy:  {lr_acc:.4f}")

# Step 5: Train XGBoost
print("\n" + "="*70)
print("6. PRIMARY MODEL: XGBOOST")
print("="*70)

# XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'random_state': 42
}

# Train model
xgb_model = xgb.XGBClassifier(**params)
xgb_model.fit(X_train, y_train, verbose=False)

# Predict on test set
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_pred_proba_xgb >= 0.5).astype(int)

# Calculate metrics
xgb_ll = log_loss(y_test, y_pred_proba_xgb)
xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb)
xgb_acc = accuracy_score(y_test, y_pred_xgb)

print(f"Test Set Results:")
print(f"  Log Loss:  {xgb_ll:.4f}")
print(f"  AUC:       {xgb_auc:.4f}")
print(f"  Accuracy:  {xgb_acc:.4f}")

# Step 6: Compare models
print("\n" + "="*70)
print("7. MODEL COMPARISON")
print("="*70)

print("\n                   Logistic Regression    XGBoost         Improvement")
print("-" * 70)

ll_improvement = ((lr_ll - xgb_ll) / lr_ll) * 100
auc_improvement = ((xgb_auc - lr_auc) / lr_auc) * 100
acc_improvement = ((xgb_acc - lr_acc) / lr_acc) * 100

print(f"Log Loss:     {lr_ll:.4f}              {xgb_ll:.4f}          {ll_improvement:+.1f}%")
print(f"AUC:          {lr_auc:.4f}              {xgb_auc:.4f}          {auc_improvement:+.1f}%")
print(f"Accuracy:     {lr_acc:.4f}              {xgb_acc:.4f}          {acc_improvement:+.1f}%")

# Step 7: Feature importance (using the trained XGBoost model)
print("\n" + "="*70)
print("8. FEATURE IMPORTANCE")
print("="*70)

importance_df = pd.DataFrame({
    'feature': available_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop features for predicting pass completion:")
for idx, row in importance_df.iterrows():
    print(f"  {row['feature']:25s} {row['importance']:.4f}")

# Step 8: Sample predictions
print("\n" + "="*70)
print("9. SAMPLE PREDICTIONS")
print("="*70)

# Get predictions for test set
sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
sample_X = X_test[sample_indices]
sample_y = y_test[sample_indices]
sample_probs = xgb_model.predict_proba(sample_X)[:, 1]

# Get corresponding play info
test_plays = play_df.iloc[np.where(np.isin(np.arange(len(play_df)), 
                                    [i for i in range(len(play_df)) if i >= len(X_train)]))[0]]
sample_plays = test_plays.iloc[sample_indices]

print("\nPredictions for sample test plays:")
print("-" * 70)
for idx, (prob, actual) in enumerate(zip(sample_probs, sample_y)):
    play = sample_plays.iloc[idx]
    actual_str = "✓ Complete" if actual == 1 else "✗ Incomplete"
    print(f"Play {play['playId']}: Down={int(play['down'])}, YardsToGo={int(play['yardsToGo'])}")
    print(f"  Predicted: {prob:.1%} completion probability")
    print(f"  Actual:    {actual_str}")
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
    pickle.dump(xgb_model, f)
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
print(f"  • Log Loss: {xgb_ll:.4f}")
print(f"  • AUC:      {xgb_auc:.4f}")
print(f"  • Accuracy: {xgb_acc:.1%}")
print(f"\nModel files saved to: {models_dir}")
print(f"Results saved to: {results_dir}")
print(f"\nNext step: Add spatial features (receiver separation, air yards, etc.)")
print(f"Expected improvement: 10-20% reduction in Log Loss")
print("="*70)