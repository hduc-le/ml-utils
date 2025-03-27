import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plotly_ts_box_plot(
    pdf, col_datetime, col_target, title, lower_bound=None, upper_bound=None
):
    # Filter data based on lower and upper bounds
    if lower_bound is not None:
        lower_bound_value = pdf[col_target].quantile(lower_bound)
        pdf = pdf[pdf[col_target] >= lower_bound_value]
    if upper_bound is not None:
        upper_bound_value = pdf[col_target].quantile(upper_bound)
        pdf = pdf[pdf[col_target] <= upper_bound_value]

    fig = go.Figure()
    fig.add_trace(go.Box(x=pdf[col_datetime], y=pdf[col_target], name=col_target))
    fig.update_layout(title=title)
    fig.show()


def plot_1d_distinct_values_over_time(
    df, time_column, value_column, title="", **kwargs
):
    # Count distinct values for each time period
    distinct_counts = df.groupby(time_column)[value_column].nunique()
    # Convert the Series to a DataFrame
    distinct_counts_df = distinct_counts.reset_index().sort_values([time_column])
    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Table", "Figure"),
        specs=[[{"type": "table"}, {"type": "xy"}]],
    )
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(distinct_counts_df.columns),
                fill_color="paleturquoise",
                align="left",
            ),
            cells=dict(
                values=[
                    distinct_counts_df[time_column],
                    distinct_counts_df[value_column],
                ],
                fill_color="lavender",
                align="left",
            ),
        ),
        row=1,
        col=1,
    )
    # Add a scatter plot to the right side of the subplot
    fig.add_trace(
        go.Scatter(
            x=distinct_counts_df[time_column],
            y=distinct_counts_df[value_column],
            mode="lines",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(**kwargs)
    fig.show()


def plot_2d_distinct_values_over_time(
    df, time_column, value_column, breakdown_column, **kwargs
):
    # Count distinct values for each time period and breakdown column
    distinct_counts = (
        df.groupby([time_column, breakdown_column])[value_column]
        .nunique()
        .reset_index()
    )

    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Table", "Figure"),
        specs=[[{"type": "table"}, {"type": "xy"}]],
    )

    # Add a table to the left side of the subplot
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(distinct_counts.columns),
                fill_color="paleturquoise",
                align="left",
            ),
            cells=dict(
                values=[
                    distinct_counts[time_column],
                    distinct_counts[breakdown_column],
                    distinct_counts[value_column],
                ],
                fill_color="lavender",
                align="left",
            ),
        ),
        row=1,
        col=1,
    )

    # Add a stacked bar chart to the right side of the subplot
    for breakdown_value in distinct_counts[breakdown_column].unique():
        fig.add_trace(
            go.Bar(
                x=distinct_counts[distinct_counts[breakdown_column] == breakdown_value][
                    time_column
                ],
                y=distinct_counts[distinct_counts[breakdown_column] == breakdown_value][
                    value_column
                ],
                name=str(breakdown_value),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(barmode="stack", **kwargs)
    fig.show()


def plot_1d_sum_values_over_time(df, time_column, value_column, title="", **kwargs):
    # Count distinct values for each time period
    distinct_counts = df.groupby(time_column)[value_column].sum()
    # Convert the Series to a DataFrame
    distinct_counts_df = distinct_counts.reset_index().sort_values([time_column])
    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Table", "Figure"),
        specs=[[{"type": "table"}, {"type": "xy"}]],
    )
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(distinct_counts_df.columns),
                fill_color="paleturquoise",
                align="left",
            ),
            cells=dict(
                values=[
                    distinct_counts_df[time_column],
                    distinct_counts_df[value_column],
                ],
                fill_color="lavender",
                align="left",
            ),
        ),
        row=1,
        col=1,
    )
    # Add a scatter plot to the right side of the subplot
    fig.add_trace(
        go.Scatter(
            x=distinct_counts_df[time_column],
            y=distinct_counts_df[value_column],
            mode="lines",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(**kwargs)
    fig.show()
