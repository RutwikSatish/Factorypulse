import pandas as pd

def default_stations() -> pd.DataFrame:
    return pd.DataFrame({
        "Station": [f"Station {i}" for i in range(1, 6)],
        "Process Time (min/unit)": [1.2, 1.0, 1.6, 1.1, 1.3],
        "CV": [0.9, 0.8, 1.1, 0.7, 0.9],
        "Availability (%)": [95, 93, 92, 96, 94],
        "MTBF (min)": [600, 500, 450, 700, 550],
        "MTTR (min)": [30, 35, 40, 25, 30],
    })

def clean_station_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Station"] = df["Station"].astype(str)

    df["Process Time (min/unit)"] = pd.to_numeric(df["Process Time (min/unit)"], errors="coerce").fillna(1.0).clip(lower=0.05)
    df["CV"] = pd.to_numeric(df["CV"], errors="coerce").fillna(1.0).clip(lower=0.0, upper=3.0)

    df["Availability (%)"] = pd.to_numeric(df["Availability (%)"], errors="coerce").fillna(95).clip(lower=50, upper=100)
    df["MTBF (min)"] = pd.to_numeric(df["MTBF (min)"], errors="coerce").fillna(600).clip(lower=10)
    df["MTTR (min)"] = pd.to_numeric(df["MTTR (min)"], errors="coerce").fillna(30).clip(lower=0.1)

    return df
