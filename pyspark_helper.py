from .log_helper import logger_func


@logger_func()
def load_bq_data_by_spark(
    spark,
    query_str,
    parent_project="momovn-mlgame-shared",
    materialization_dataset="team_ml_analytics",
    materialization_project="project-5400504384186300846",
    num_partitions=20,
):
    df01 = (
        spark.read.format("bigquery")
        .option("viewsEnabled", "true")
        .option("parentProject", parent_project)
        .option("materializationDataset", materialization_dataset)
        .option("materializationProject", materialization_project)
        .load(query_str)
        .coalesce(num_partitions)
    )
    return df01
