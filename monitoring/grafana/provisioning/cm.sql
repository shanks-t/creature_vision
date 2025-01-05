WITH parsed_matrix AS (
  SELECT 
    model_version,
    timestamp,
    SPLIT(cm_entry, ':') as parts
  FROM dog_prediction_app.model_metrics,
  UNNEST(SPLIT(confusion_matrix, '|')) as cm_entry
)
SELECT 
  model_version,
  timestamp,
  parts[OFFSET(0)] as true_label,
  parts[OFFSET(1)] as predicted_label,
  CAST(parts[OFFSET(2)] as INT64) as count
FROM parsed_matrix
WHERE timestamp BETWEEN @start_time AND @end_time
