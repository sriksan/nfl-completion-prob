import pandas as pd

DATA_PATH = "data/raw/"

def load_games():
    """Load games.csv"""
    return pd.read_csv(DATA_PATH + "games.csv")

def load_players():
    """Load players.csv"""
    return pd.read_csv(DATA_PATH + "players.csv")

def load_plays():
    """
    Load plays.csv and filter to pass plays only
    
    Pass plays are identified by having a non-null PassResult
    PassResult values:
    - 'C' = Complete
    - 'I' = Incomplete  
    - 'IN' = Interception
    - 'S' = Sack
    - 'R' = Run from QB
    """
    plays = pd.read_csv(DATA_PATH + "plays.csv")
    
    # Keep only pass plays
    pass_plays = plays[plays['PassResult'].notna()].copy()
    
    # Pandas DataFrame (Binary column for completion)
    # Complete = 1, Everything else = 0
    pass_plays['complete'] = (pass_plays['PassResult'] == 'C').astype(int)
    
    print(f"Total plays: {len(plays):,}")
    print(f"Pass plays: {len(pass_plays):,}")
    print(f"Completion rate: {pass_plays['complete'].mean():.1%}")
    
    return pass_plays

def load_tracking(week=1):
    """
    Load tracking data for specific week
    """
    return pd.read_csv(DATA_PATH + f"tracking_week{week}.csv")

def load_pass_forward_frames():
    """
    Load tracking data and filter to only pass_forward events
    """
    tracking = load_tracking(week=1)
    pass_forward = tracking[tracking['event'] == 'pass_forward'].copy()
    
    print(f"Total tracking frames: {len(tracking):,}")
    print(f"Pass forward frames: {len(pass_forward):,}")
    print(f"Unique plays: {pass_forward['playId'].nunique()}")
    
    return pass_forward

def load_all_data():
    """
    function to load everything at once
    """
    games = load_games()
    players = load_players()
    plays = load_plays()
    tracking = load_tracking()
    
    return {
        'games': games,
        'players': players,
        'plays': plays,
        'tracking': tracking
    }

if __name__ == "__main__":
    # Test the loading functions
    print("Testing data loading...")
    print("="*60)
    
    plays = load_plays()
    print("\nPass result distribution:")
    print(plays['PassResult'].value_counts())
    
    print("\n" + "="*60)
    pass_forward = load_pass_forward_frames()