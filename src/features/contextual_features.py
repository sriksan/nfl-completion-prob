def add_contextual_features(df):
    """
    game context features
    """
    
    # Down indicators
    df['is_first_down'] = (df['down'] == 1).astype(int)
    df['is_second_down'] = (df['down'] == 2).astype(int)
    df['is_third_down'] = (df['down'] == 3).astype(int)
    df['is_fourth_down'] = (df['down'] == 4).astype(int)
    
    # Distance indicators
    df['is_short_yardage'] = (df['yardsToGo'] <= 3).astype(int)
    df['is_medium_yardage'] = ((df['yardsToGo'] > 3) & (df['yardsToGo'] < 10)).astype(int)
    df['is_long_yardage'] = (df['yardsToGo'] >= 10).astype(int)
    
    # Combined situation
    df['is_third_and_long'] = ((df['down'] == 3) & (df['yardsToGo'] >= 7)).astype(int)

    print(f"Added {8} contextual features")
    
    return df