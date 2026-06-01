import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)

    df["Start"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Start Time"].astype(str),
        format="mixed",
        errors="coerce"
    )

    df = df.dropna(subset=["Start", "Duration (minutes)"])
    df = df[df["Duration (minutes)"] > 0]

    df["Hour"] = df["Start"].dt.hour
    df = df.sort_values(by=["Person ID", "Start"]).reset_index(drop=True)
    return df