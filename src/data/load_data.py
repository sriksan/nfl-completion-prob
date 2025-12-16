import pandas as pd

DATA_PATH = "data/raw/"

def load_plays():
    plays = pd.read_csv(DATA_PATH + "plays.csv")

    # Keep only pass plays
    pass_plays = plays[
        plays["playType"] == "pass"
    ].copy()

    # Binary label
    pass_plays["complete"] = (pass_plays["passResult"] == "C").astype(int)

    return pass_plays
