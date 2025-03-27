from pyspark.sql import functions as F
from pyspark.sql.window import Window


# Function to assign tag to each row in PySpark
def assign_tag(df, score_col="prob", tag_col="tag", n=10):
    # Create a window for ranking based on score
    window_spec = Window.orderBy(F.col(score_col).desc())

    # Calculate the rank of each row, then divide into bins
    df = df.withColumn(
        "rank", F.row_number().over(window_spec)
    )  # Rank rows based on score
    total_count = df.count()
    bin_size = total_count // n

    # Assign tag based on the rank
    df = df.withColumn(tag_col, (F.col("rank") / bin_size + 1).cast("int"))
    df = df.withColumn(
        tag_col, F.least(F.col(tag_col), F.lit(n))
    )  # Ensure tag does not exceed 'n'

    # Drop the rank column as it's no longer needed
    df = df.drop("rank")
    return df


# Function to calculate precision, recall by tag in PySpark with rounded values
def get_pr_rc_by_tag(df, target_col="flag_bad", tag_col="tag"):
    # Aggregate to get count and sum of positives for each tag
    result = (
        df.groupBy(tag_col)
        .agg(
            F.count(target_col).alias("n_total"),
            F.sum(F.col(target_col)).alias("n_pos"),
        )
        .orderBy(tag_col)
    )

    # Calculate cumulative sums
    window_spec = Window.orderBy(tag_col).rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )
    result = result.withColumn(
        "n_total_cumsum", F.sum("n_total").over(window_spec)
    )
    result = result.withColumn(
        "n_pos_cumsum", F.sum("n_pos").over(window_spec)
    )

    # Calculate total number of samples and positives
    total_samples = result.select(F.sum("n_total")).collect()[0][0]
    total_positives = result.select(F.sum("n_pos")).collect()[0][0]

    # Calculate percentages and metrics with rounding to 4 decimal places
    result = result.withColumn(
        "n_total_perc", F.round(F.col("n_total") / total_samples * 100, 3)
    )
    result = result.withColumn(
        "n_pos_perc", F.round(F.col("n_pos") / total_positives * 100, 3)
    )
    result = result.withColumn(
        "prec_perc", F.round(F.col("n_pos") / F.col("n_total") * 100, 3)
    )
    result = result.withColumn(
        "prec_cumsum_perc",
        F.round(F.col("n_pos_cumsum") / F.col("n_total_cumsum") * 100, 3),
    )
    result = result.withColumn(
        "recall_perc", F.round(F.col("n_pos") / total_positives * 100, 3)
    )
    result = result.withColumn(
        "recall_cumsum_perc",
        F.round(F.col("n_pos_cumsum") / total_positives * 100, 3),
    )

    return result
