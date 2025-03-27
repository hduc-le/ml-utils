import pandas as pd


def train_test_split_out_of_time(
    df: pd.DataFrame, datetime_column: str, train_fraction: float = 0.8
):
    df = df.sort_values(by=datetime_column)
    date_counts = df.groupby(datetime_column).size().reset_index(name="count")
    date_counts["cumulative_count"] = date_counts["count"].cumsum()
    total_samples = df.shape[0]
    train_size = int(total_samples * train_fraction)
    split_date = date_counts[date_counts["cumulative_count"] >= train_size][
        datetime_column
    ].iloc[0]
    train_df = df[df[datetime_column] < split_date]
    test_df = df[df[datetime_column] >= split_date]
    return train_df, test_df
