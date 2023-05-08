from diffusers import DiffusionPipeline, EulerDiscreteScheduler

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
pipeline
