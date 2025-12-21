import pandas as pd
from load_data import load_pass_forward_frames, load_plays, load_players

def merge_tracking_with_plays(tracking_df, plays_df):
    """
    Merge tracking data with play-level context
    
    This adds contextual information to each tracking frame:
    - down, yardsToGo (game situation)
    - quarter, GameClock (time context)
    - HomeScoreBeforePlay, VisitorScoreBeforePlay (score)
    - PassResult (target variable)
    - complete (binary target: 1 or 0)
    """
    print(f"Tracking rows before merge: {len(tracking_df):,}")
    print(f"Plays rows: {len(plays_df):,}")
    
    merged = tracking_df.merge(
        plays_df,
        on=['gameId', 'playId'],
        how='left'
    )
    
    print(f"Tracking rows after merge: {len(merged):,}")
    print(f"New columns added: {len(merged.columns) - len(tracking_df.columns)}")
    
    # Check for null targets
    null_targets = merged['complete'].isna().sum()
    if null_targets > 0:
        print(f"Warning: {null_targets} rows have null 'complete' values")
    
    return merged

def merge_tracking_with_players(tracking_df, players_df):
    """
    Add player metadata to tracking data
    
    Note: Tracking data already has position and displayName,
    so this function might not be needed. Keeping for future use.
    """
    print(f"Skipping player merge - tracking already has player info")
    return tracking_df

def create_modeling_dataset():
    """
    Create the complete dataset for modeling
    
    Process:
    1. Load pass_forward frames (moment ball is thrown)
    2. Merge with plays.csv (add context: down, distance, score, target)
    3. Merge with players.csv (add player positions)
    
    Returns:
        DataFrame with one row per player per pass_forward event
        Each row has: tracking data + game context + player info + target
    """
    print("="*60)
    print("CREATING MODELING DATASET")
    print("="*60)
    
    # Load pass_forward frames
    print("\n1. Loading pass_forward frames...")
    pass_forward = load_pass_forward_frames()
    
    # Add play context
    print("\n2. Merging with plays data...")
    plays = load_plays()
    merged = merge_tracking_with_plays(pass_forward, plays)
    
    # Add player info
    print("\n3. Merging with players data...")
    players = load_players()
    merged = merge_tracking_with_players(merged, players)
    
    print("\n" + "="*60)
    print("DATASET CREATION COMPLETE")
    print("="*60)
    print(f"Total rows: {len(merged):,}")
    print(f"Total columns: {len(merged.columns)}")
    print(f"Unique plays: {merged['playId'].nunique()}")
    print(f"\nTarget distribution:")
    print(merged.groupby('playId')['complete'].first().value_counts())
    print(f"Completion rate: {merged.groupby('playId')['complete'].first().mean():.1%}")
    
    # Show sample of columns
    print(f"\nKey columns available:")
    key_cols = ['gameId', 'playId', 'nflId', 'displayName', 'position', 
                'x', 'y', 's', 'a', 'down', 'yardsToGo', 'quarter', 
                'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 
                'PassResult', 'complete']
    existing = [col for col in key_cols if col in merged.columns]
    for col in existing:
        print(f"  ✓ {col}")
    
    return merged

def get_pass_forward_by_play(merged_df, game_id, play_id):
    """
    Helper function to get all tracking data for a specific play
    Useful for debugging and visualization
    """
    play_data = merged_df[
        (merged_df['gameId'] == game_id) & 
        (merged_df['playId'] == play_id)
    ].copy()
    
    print(f"\nPlay: gameId={game_id}, playId={play_id}")
    print(f"Players on field: {len(play_data)}")
    print(f"Down: {play_data['down'].iloc[0]}")
    print(f"Yards to go: {play_data['yardsToGo'].iloc[0]}")
    print(f"Result: {play_data['PassResult'].iloc[0]}")
    print(f"Complete: {play_data['complete'].iloc[0]}")
    
    return play_data

if __name__ == "__main__":
    df = create_modeling_dataset()
    
    #sample play
    print("\n" + "="*60)
    print("SAMPLE PLAY")
    print("="*60)
    sample_game = df['gameId'].iloc[0]
    sample_play = df['playId'].iloc[0]
    
    sample = get_pass_forward_by_play(df, sample_game, sample_play)
    print("\nSample rows:")
    cols_to_show = []
    for col in ['displayName', 'position', 'team', 'x', 'y', 's']:
        if col in sample.columns:
            cols_to_show.append(col)

    print(sample[cols_to_show].head(10))