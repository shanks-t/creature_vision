from .dataflow import ProcessDataPipeline


pipeline = ProcessDataPipeline('creature-vision-training-set', batch_size=320)
pipeline.run_pipeline()
