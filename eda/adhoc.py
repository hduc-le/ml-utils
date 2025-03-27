import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compare_distribution(pdf01_train, pdf01_val, pdf01_test, cname_target, nbins=100):
    # Create a subplot with 3 rows and 1 column
    fig = make_subplots(rows=3, cols=1)

    # Add histogram of target variable for training data to the subplot
    fig.add_trace(
        go.Histogram(x=pdf01_train[cname_target], nbinsx=nbins, name="Train"),
        row=1,
        col=1,
    )

    # Add histogram of target variable for validation data to the subplot
    fig.add_trace(
        go.Histogram(x=pdf01_val[cname_target], nbinsx=nbins, name="Validation"),
        row=2,
        col=1,
    )

    # Add histogram of target variable for test data to the subplot
    fig.add_trace(
        go.Histogram(x=pdf01_test[cname_target], nbinsx=nbins, name="Test"),
        row=3,
        col=1,
    )

    # Update layout for better visualization
    fig.update_layout(
        height=600, width=800, title_text="Distribution of Target Variable"
    )
    fig.show()


### Acurracy Rate
def evaluate_predictions(
    pdf_train, pdf_val, pdf_test, cname_target, predicted_col="predicted_values"
):
    # Define function to calculate accuracy rate
    def accuracy_rate(forecast, actual, lower_bound_multiplier, upper_bound_multiplier):
        lower_bound = lower_bound_multiplier * actual
        upper_bound = upper_bound_multiplier * actual
        if lower_bound <= forecast <= upper_bound:
            return 1
        else:
            return 0

    # Calculate accuracy for each range and store results in dataframes
    df_records = pd.DataFrame()
    df_accuracy = pd.DataFrame()

    for i in range(5, 55, 5):  # thresholds from 5% to 50% with interval of 5%
        lower_bound_multiplier = 1 - i / 100
        upper_bound_multiplier = 1 + i / 100

        for df, df_name in zip(
            [pdf_train, pdf_val, pdf_test], ["Train", "Val", "Test"]
        ):
            df[f"accuracy_{i}"] = df.apply(
                lambda x: accuracy_rate(
                    x[predicted_col],
                    x[cname_target],
                    lower_bound_multiplier,
                    upper_bound_multiplier,
                ),
                axis=1,
            )
            df_records.loc[df_name, f"{i}%"] = df[f"accuracy_{i}"].sum()
            df_accuracy.loc[df_name, f"{i}%"] = 100 * df[f"accuracy_{i}"].mean()

    return df_records, df_accuracy


#
def accuracy_over_breakdown_col(
    pdf_train, cname_target, col_breakdown="etl_date", predicted_col="predicted_values"
):
    # Define function to calculate accuracy rate
    def accuracy_rate(forecast, actual, lower_bound_multiplier, upper_bound_multiplier):
        lower_bound = lower_bound_multiplier * actual
        upper_bound = upper_bound_multiplier * actual
        if lower_bound <= forecast <= upper_bound:
            return 1
        else:
            return 0

    # Initialize an empty dataframe to store results
    df_accuracy = pd.DataFrame()
    for i in range(5, 55, 5):  # thresholds from 5% to 50% with interval of 5%
        lower_bound_multiplier = 1 - i / 100
        upper_bound_multiplier = 1 + i / 100
        # Calculate accuracy for each row
        pdf_train[f"accuracy_{i}"] = pdf_train.apply(
            lambda x: accuracy_rate(
                x[predicted_col],
                x[cname_target],
                lower_bound_multiplier,
                upper_bound_multiplier,
            ),
            axis=1,
        )
        # Group by etl_date and calculate mean accuracy for each group
        accuracy_over_time = (
            100 * pdf_train.groupby(col_breakdown)[f"accuracy_{i}"].mean()
        )
        # Store the results in the dataframe
        df_accuracy[f"{i}%"] = accuracy_over_time
    df_sample = pd.DataFrame(pdf_train.groupby(col_breakdown).size())
    return df_accuracy, df_sample
