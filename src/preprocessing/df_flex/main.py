import argparse

from df_flex.pipeline import ProcessDataPipeline
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="v3_5")

    # Use local flag to override to DirectRunner
    parser.add_argument(
        "--local",
        "-l",
        action="store_true",
        help="Run pipeline locally with DirectRunner",
    )

    # Parse known args first, then pass remaining args to PipelineOptions
    known_args, _ = parser.parse_known_args(argv)

    use_dataflow = not known_args.local  # Default is Dataflow unless --local is set

    pipeline = ProcessDataPipeline(
        dataset_bucket_name="creature-vision-training-set",
        dataflow_bucket_name="dataflow-use1",
        project_id="creature-vision",
    )

    pipeline.run_pipeline(
        version=known_args.version,
        use_dataflow=use_dataflow,
        region="us-east1",
        random_seed=known_args.random_seed,
        max_num_workers=2,
        number_of_worker_harness_threads=4,
        machine_type="n1-standard-2",
    )


if __name__ == "__main__":
    run()
