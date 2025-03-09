import argparse

from df_flex.pipeline import ProcessDataPipeline
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions


def run(argv=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_files',
        type=int,
        default=500,
        help='Maximum number of files to process'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--version',
        type=str,
        default="",
        help='version of the training run equal to date'
    )

    # Parse known args first, then pass remaining args to PipelineOptions
    known_args, pipeline_args = parser.parse_known_args(argv)

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    pipeline = ProcessDataPipeline(
        dataset_bucket_name='creature-vision-training-set',
        dataflow_bucket_name='dataflow-use1',
        project_id='creature-vision'
    )

    pipeline.run_pipeline(
        version=known_args.version,
        use_dataflow=True,
        region=pipeline_options.get_all_options().get('region', 'us-east1'),
        max_files=known_args.max_files,
        random_seed=known_args.random_seed,
        max_num_workers=2,
        number_of_worker_harness_threads=4,
        machine_type='n1-standard-2'
    )


if __name__ == '__main__':
    run()
