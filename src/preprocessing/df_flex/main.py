from df_flex.pipeline import ProcessDataPipeline
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions


def run(argv=None):
    pipeline_options = PipelineOptions(argv)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    pipeline = ProcessDataPipeline(
        dataset_bucket_name='creature-vision-training-set',
        dataflow_bucket_name='dataflow-use1',
        project_id='creature-vision'
    )

    pipeline.run_pipeline(
        use_dataflow=True,
        region=pipeline_options.get_all_options().get('region', 'us-east1'),
        max_files=1200,
        random_seed=420,
        max_num_workers=2,
        number_of_worker_harness_threads=4,
        machine_type='n1-standard-2'
    )


if __name__ == '__main__':
    run()
