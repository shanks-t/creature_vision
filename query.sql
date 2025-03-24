WITH misclassifications AS (
  SELECT
    model_version,
    actual,
    predicted,
    COUNT(*) AS count
  FROM
    `creature-vision.dog_prediction_app.prediction_metrics`
  WHERE
    actual != predicted
  GROUP BY
    model_version, actual, predicted
),
ranked_misclassifications AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY model_version
      ORDER BY count DESC
    ) AS rank
  FROM misclassifications
)
SELECT
  model_version,
  actual,
  predicted,
  count
FROM
  ranked_misclassifications
WHERE
  rank <= 10
ORDER BY
  model_version,
  count DESC;
