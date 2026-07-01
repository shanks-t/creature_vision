import argparse
import os

from df_flex.pipeline import ProcessDataPipeline


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="v3_5")
    parser.add_argument(
        "--dataset_bucket",
        default=os.getenv("DATA_BUCKET_NAME", "creature-vision-training-set"),
    )
    parser.add_argument(
        "--dataflow_bucket",
        default=os.getenv("DATAFLOW_BUCKET_NAME", "dataflow-use1"),
    )
    parser.add_argument(
        "--project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT", "creature-vision"),
    )
    parser.add_argument("--region", default=os.getenv("CLOUD_LOCATION", "us-east1"))

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
        dataset_bucket_name=known_args.dataset_bucket,
        dataflow_bucket_name=known_args.dataflow_bucket,
        project_id=known_args.project,
    )

    pipeline.run_pipeline(
        version=known_args.version,
        use_dataflow=use_dataflow,
        region=known_args.region,
        random_seed=known_args.random_seed,
        max_files=known_args.max_files,
        max_num_workers=2,
        number_of_worker_harness_threads=4,
        machine_type="n1-standard-2",
    )


if __name__ == "__main__":
    run()
