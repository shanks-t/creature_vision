import pytest
from datetime import datetime

from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
import apache_beam as beam


from src.preprocessing.df_flex.pipeline import FormatLabelStatsForBigQuery


def test_format_label_stats_for_bigquery():
    model_version = 'v1'

    input_data = [
        (('pitbull', 0), 120),
        (('doberman', 1), 80)
    ]

    expected = [
        {
            'model_version': 'v1',
            'class_label': 'pitbull',
            'label_id': 0,
            'example_count': 120,
            'run_timestamp': 'ANY'  # Placeholder for timestamp
        },
        {
            'model_version': 'v1',
            'class_label': 'doberman',
            'label_id': 1,
            'example_count': 80,
            'run_timestamp': 'ANY'
        }
    ]

    with TestPipeline() as p:
        output = (
            p
            | 'CreateInput' >> beam.Create(input_data)
            | 'FormatForBQ' >> beam.ParDo(FormatLabelStatsForBigQuery(model_version))
        )

        def check_format(actual):
            # Strip run_timestamp for testing purposes
            simplified = [
                {k: (v if k != 'run_timestamp' else 'ANY')
                 for k, v in row.items()}
                for row in actual
            ]
            assert simplified == expected

        assert_that(output, check_format)
