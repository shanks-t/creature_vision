from mcp.server.fastmcp import FastMCP
from google.cloud import bigquery

import sys

print("Running Python from:", sys.executable, file=sys.stderr)


# Optional: set your default project here
PROJECT_ID = "creature-vision"
DATASET = "dog_prediction_app"
TABLE = "prediction_metrics"

mcp = FastMCP("PredictionQueryBot")

# Initialize BigQuery client
bq_client = bigquery.Client(project=PROJECT_ID)


@mcp.tool(
    name="get_top_breeds_by_accuracy",
    description="Return top N dog breeds with highest prediction accuracy for a given model version",
)
def get_top_breeds_by_accuracy(model_version: str, limit: int = 5) -> list[dict]:
    query = f"""
        SELECT actual AS breed, 
               COUNT(*) AS total, 
               SUM(IF(is_correct, 1, 0)) AS correct,
               SAFE_DIVIDE(SUM(IF(is_correct, 1, 0)), COUNT(*)) AS accuracy
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        WHERE model_version = @model_version
        GROUP BY breed
        ORDER BY accuracy DESC
        LIMIT @limit
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("model_version", "STRING", model_version),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ]
    )
    result = bq_client.query(query, job_config=job_config).result()
    return [dict(row) for row in result]


@mcp.tool(
    name="get_dog_prediction_accuracy",
    description="Return accuracy for a given model version and dog breed from BigQuery",
)
def get_dog_prediction_accuracy(model_version: str, breed: str) -> float:
    """
    Queries BigQuery to get accuracy for a specific dog breed and model version.
    """
    query = f"""
        SELECT 
            AVG(IF(is_correct, 1.0, 0.0)) AS accuracy
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        WHERE model_version = @model_version AND actual = @breed
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("model_version", "STRING", model_version),
            bigquery.ScalarQueryParameter("breed", "STRING", breed),
        ]
    )
    result = bq_client.query(query, job_config=job_config).result()
    row = next(result, None)
    return round(row.accuracy * 100, 2) if row and row.accuracy is not None else 0.0
