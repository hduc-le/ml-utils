# ml-utils

A collection of Python utilities for machine learning workflows, covering data ingestion, preprocessing, evaluation, and experiment tracking on GCP infrastructure.

## Modules

---

### `bigquery_helper.py`

Utilities for interacting with Google BigQuery.

```python
from bigquery_helper import BigqueryHelper

bq = BigqueryHelper(project_id="my-project", dataset_name="my_dataset")

# Inspect a table
print(bq.list_tables())
print(bq.table_schema("my-project.my_dataset.my_table"))
print(bq.head("my-project.my_dataset.my_table", num_rows=10))

# Check cost before running
gb = bq.estimate_query_size("SELECT * FROM my_dataset.events WHERE date = '2024-01-01'")
print(f"Estimated scan: {gb:.2f} GB")

# Run queries
df = bq.query_to_dataframe("SELECT user_id, amount FROM my_dataset.transactions LIMIT 1000")

# Safe query — aborts if scan > 5 GB
df = bq.query_to_dataframe_safe("SELECT * FROM my_dataset.big_table", max_gb_scanned=5)

# Stream large results without loading everything into memory
for chunk in bq.query_to_dataframe_iterable("SELECT * FROM my_dataset.huge_table"):
    process(chunk)

# Materialize query results
bq.query_to_table("SELECT * FROM src WHERE dt = '2024-01-01'", "my-project.my_dataset.snapshot")
bq.query_to_csv("SELECT * FROM my_dataset.results", "gs://my-bucket/exports/results_*.csv")

# Write a DataFrame back to BQ
bq.write_to_table(df, "my-project.my_dataset.predictions", if_exists="replace")

# Cleanup
bq.delete_table("my-project.my_dataset.old_table")
```

---

### `storage_helper.py`

OOP wrapper around the GCS Python client.

```python
from storage_helper import GCSHelper

gcs = GCSHelper(bucket_name="my-bucket")

# Upload
gcs.upload_file("local/model.pkl", "models/v1/model.pkl")
gcs.upload_from_string('{"threshold": 0.5}', "configs/threshold.json", content_type="application/json")

# Download
gcs.download_file("models/v1/model.pkl", "local/model.pkl")
config_text = gcs.download_file_as_text("configs/threshold.json")

# Inspect
print(gcs.list_blobs(prefix="models/v1/"))
print(gcs.file_exists("models/v1/model.pkl"))

# Delete
gcs.delete_blob("models/v1/model.pkl")
gcs.delete_folder("models/v1/")   # deletes all blobs under the prefix
```

---

### `file_helper.py`

Functional helpers for GCS and local file I/O.

```python
from file_helper import read_yaml, read_sql, upload_from_filename, gcs_get_last_file_name
from google.cloud import storage

client = storage.Client()

# Read configs from local or GCS transparently
config = read_yaml("configs/train.yaml")
config = read_yaml("gs://my-bucket/configs/train.yaml")

# Read a SQL file
query = read_sql("sql/feature_query.sql")

# Upload a file
upload_from_filename(client, "outputs/model.pkl", "gs://my-bucket/models/model.pkl")

# Get the latest file in a GCS directory (e.g. latest daily snapshot)
latest = gcs_get_last_file_name(client, "my-bucket", "data/snapshots/")
# → "gs://my-bucket/data/snapshots/2024-01-31.parquet"
```

---

### `mlflow_helper.py`

Wrapper around MLflow for experiment tracking.

```python
from mlflow_helper import MLflowHelper

mlflow = MLflowHelper(
    experiment_name="credit-scoring-v2",
    tracking_url="http://mlflow.internal:5000",
    artifact_location="gs://my-bucket/mlflow",
)

with mlflow.start_run(run_name="lgbm-baseline"):
    mlflow.log_params({"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6})
    mlflow.log_metrics({"auc": 0.82, "log_loss": 0.41, "ece": 0.03})

    mlflow.log_dataframe(feature_importance_df, name="feature_importance")
    mlflow.log_yaml({"features": feature_list, "target": "label"}, name="feature_config")
    mlflow.log_plot(args=(y_true, y_pred), plot_func=plot_roc_curve, fp="roc_curve.html")

# Retrieve artifacts from the latest run for serving
model_uri = mlflow.get_model_uri_from_latest_run(
    run_name="lgbm-baseline",
    experiment_name="credit-scoring-v2",
)
# → "gs://my-bucket/mlflow/.../artifacts/model"
```

---

### `pyspark_helper.py`

Spark utility for loading BigQuery data.

```python
from pyspark_helper import load_bq_data_by_spark

df = load_bq_data_by_spark(
    spark,
    query_str="SELECT * FROM my_dataset.features WHERE dt = '2024-01-01'",
    parent_project="my-gcp-project",
    materialization_dataset="team_ml_staging",
    num_partitions=40,
)
```

---

### `log_helper.py`

Decorator for function-level timing and logging.

```python
from log_helper import logger_func

@logger_func(call_depth=0)
def train_model(X, y):
    ...

@logger_func(call_depth=0)
def run_pipeline(X, y):
    model = train_model(X, y)   # nested call is indented in logs
    ...

# Logs emitted:
# ----- run_pipeline: start AT 2024-01-01 10:00:00 -----
#  ----- train_model: start AT 2024-01-01 10:00:00 -----
#  ----- train_model: end took 12.345678s | 2024-01-01 10:00:12 -----
# ----- run_pipeline: end took 13.001234s | 2024-01-01 10:00:13 -----
```

---

## `metrics/`

### `binary_classification.py`

```python
from metrics.binary_classification import compute_metrics, plot_roc_curve, plot_calibration_curve_with_count
from sklearn.metrics import roc_curve

# All standard metrics in one call
metrics = compute_metrics(y_true, y_pred_proba, prefix="val")
# → {"val_auc": 0.82, "val_log_loss": 0.41, "val_auc_pr": 0.61, "val_ece": 0.03}

# ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
fig = plot_roc_curve(fpr, tpr, auc_score=metrics["val_auc"])
fig.show()

# Calibration curve with sample counts per bin
fig = plot_calibration_curve_with_count(y_true, y_pred_proba, n_bin=10)
fig.show()
```

### `binary_metrics.py`

Decile-based lift analysis — useful for ranking model quality.

```python
from metrics.binary_metrics import assign_tag, get_pr_rc_by_tag

df = assign_tag(predictions_df, score_col="prob", tag_col="decile", n=10)
report = get_pr_rc_by_tag(df, cname_target="label", tag_col="decile")

print(report[["decile", "Precision (%)", "Recall (%)", "Precision_acc (%)", "Recall_acc (%)"]])
# decile  Precision (%)  Recall (%)  Precision_acc (%)  Recall_acc (%)
#      1          62.3        31.2               62.3            31.2
#      2          48.1        24.0               55.2            55.2
#    ...
```

---

## `eda/`

### `psi.py`

Monitor feature distribution shift between training and production data.

```python
from eda.psi import calculate_psi
import numpy as np

train_scores = np.array([...])
prod_scores  = np.array([...])

# Single variable
psi = calculate_psi(train_scores, prod_scores, buckettype="quantiles", buckets=10)
# psi < 0.1  → stable
# psi < 0.2  → minor shift
# psi >= 0.2 → significant shift, investigate

# Multiple variables at once (columns = variables)
X_train = df_train[feature_cols].values
X_prod  = df_prod[feature_cols].values
psi_values = calculate_psi(X_train, X_prod, buckettype="quantiles", buckets=10, axis=0)
# → array of PSI per feature
```

---

## `sklearn_pipeline/`

### `base_feats_generator.py`

sklearn-compatible transformers, composable with `sklearn.pipeline.Pipeline`.

```python
from sklearn.pipeline import Pipeline
from sklearn_pipeline.base_feats_generator import (
    TargetEncoder,
    BucketTargetEncoder,
    LogTransformer,
    DropColumnsTransformer,
    ColumnListTypeFeatureGenerator,
)

pipeline = Pipeline([
    ("log",    LogTransformer(columns=["amount", "balance"])),
    ("bucket", BucketTargetEncoder(columns=["credit_score", "tenure_days"])),
    ("target", TargetEncoder(columns=["city", "product_category"])),
    ("drop",   DropColumnsTransformer(columns=["raw_id", "created_at"])),
])

X_train_transformed = pipeline.fit_transform(X_train, y_train)
X_test_transformed  = pipeline.transform(X_test)
```

**`ColumnListTypeFeatureGenerator`** — for list-type columns (e.g. past product IDs):

```python
from sklearn_pipeline.base_feats_generator import ColumnListTypeFeatureGenerator

# X["past_products"] = [["loan", "savings"], ["savings"], ["loan", "insurance", "savings"], ...]
gen = ColumnListTypeFeatureGenerator(id_col="past_products", top_n=5)
X_out = gen.fit_transform(X_train)
# Adds columns: past_products_loan, past_products_savings, ..., past_products_other
```

---

### `forecast_feats_generator.py`

Specialized transformers for time-aware and ad-format feature engineering.

```python
from sklearn_pipeline.forecast_feats_generator import (
    DaysTargetEncoder,
    BudgetDaysTargetEncoder,
    DateTimeTargetEncoderCustom,
    CleanStringTransformer,
)

pipeline = Pipeline([
    ("clean",    CleanStringTransformer(columns=["campaign_name", "ad_group"])),
    ("days",     DaysTargetEncoder("start_date", "end_date", "campaign_duration")),
    ("budget",   BudgetDaysTargetEncoder("budget", "start_date", "end_date", "budget_rate")),
    ("datetime", DateTimeTargetEncoderCustom("impression_time", "ad_format")),
])

X_out = pipeline.fit_transform(X_train, y_train)
# Adds: campaign_duration (target-encoded bin), budget_num_days_rate_bin,
#       format_DOW_target, format_DOM_target
```

---

## `pyspark_pipeline/`

### `preprocessing.py`

End-to-end PySpark preprocessing — imputation, categorical encoding, and feature vector assembly — with save/load support.

```python
from pyspark_pipeline.preprocessing import PySparkPreprocessingPipeline

feature_cols = ["age", "income", "city", "product", "days_since_last_txn"]

pipeline = PySparkPreprocessingPipeline(
    feature_names=feature_cols,
    feature_column="features",
    numerical_default=-9999,
    categorical_default="unknown",
)

# Fit and transform training data
train_df = pipeline.fit_transform(spark_train_df)

# Transform new data using the same fitted encodings
test_df = pipeline.transform(spark_test_df)

# Inspect learned categorical mappings
print(pipeline.metadata["categorical_mappings"])
# → {"city": {"hanoi": 0, "hcmc": 1, "unknown": -9999, ...}, "product": {...}}

# Persist to GCS
pipeline.save("gs://my-bucket/pipelines/preprocessing_v1")

# Reload in a serving job
pipeline = PySparkPreprocessingPipeline.load("gs://my-bucket/pipelines/preprocessing_v1")
serving_df = pipeline.transform(spark_serving_df)
```

---

## `transformer/`

### `imputer.py`

PySpark ML-compatible imputers, usable standalone or inside a `pyspark.ml.Pipeline`.

```python
from transformer.imputer import ConstantImputer, NoOpTransformer
from pyspark.ml import Pipeline

num_imputer = ConstantImputer(inputCols=["age", "income"], defaultValue=-9999)
cat_imputer = ConstantImputer(inputCols=["city", "product"], defaultValue="unknown")

# Falls back to no-op when inputCols or defaultValue is None
noop = ConstantImputer()   # safe to include unconditionally

pipeline = Pipeline(stages=[cat_imputer, num_imputer])
model = pipeline.fit(df)
df_clean = model.transform(df)
```
