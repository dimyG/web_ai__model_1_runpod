import runpod
import os
import io
import base64
from ml import load_pipeline, img_from_prompt
from exceptions import ModelNotFound
# from main import generate_img

# We read the defined cache dir from the env file, and then we use the cache_dir parameter to use it.
# Using the cache_dir parameter could be avoided if we set the HUGGINGFACE_HUB_CACHE environment variable on the shell.
# But it doesn't work just by adding it in the env file, so we read it and then use it as the cache_dir parameter.
huggingface_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")

model_id_1 = "CompVis/stable-diffusion-v1-4"
model_id_2 = "stabilityai/stable-diffusion-2"
model_id_3 = "stabilityai/stable-diffusion-2-1"
# Runpod advice: load your model(s) into vram here.
# You need enough vram size to load all your models into it.
pipe_initial_1 = load_pipeline(model_id_1, cache_dir=huggingface_hub_cache)
# pipe_initial_2 = load_pipeline(model_id_2, cache_dir=huggingface_hub_cache)
pipe_initial_3 = load_pipeline(model_id_3, cache_dir=huggingface_hub_cache)


def pipeline_select(event_model):
    if event_model == model_id_1:
        pipe = pipe_initial_1
    # elif event_model == model_id_2:
    #     pipe = pipe_initial_2
    elif event_model == model_id_3:
        pipe = pipe_initial_3
    else:
        raise ModelNotFound("Model not found")
    return pipe


def handler(event):
    """
    event payload example
    {
        'delayTime': 2534,
        'id': '2a16b881-830f-4d14-af5b-f7db7c0a96fc',
        'input': {
            'prompt': 'some text.'
        },
        'status': 'IN_PROGRESS'
    }
    """
    print(event)

    prompt = event['input']['prompt']
    model = event['input']['model']
    seed = int(event['input']['seed'])
    height = int(event['input']['height'])
    width = int(event['input']['width'])
    guidance_scale = float(event['input']['guidance_scale'])
    num_inference_steps = int(event['input']['num_inference_steps'])

    selected_pipeline = pipeline_select(model)

    # Runpod requirement: If the image is larger than 2MB, store it in a blob store and return an url to it.
    image = img_from_prompt(prompt=prompt, pipe=selected_pipeline, seed=seed, height=height, width=width,
                            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
    memory_stream = io.BytesIO()
    image.save(memory_stream, format='PNG')  # write the image to memory instead of the disk, and return it from there
    memory_stream.seek(0)
    # Convert the contents of the memory stream to a base64 encoded string
    encoded_string = base64.b64encode(memory_stream.getvalue()).decode('utf-8')
    return encoded_string


# todo: watch the source code and restart the server on changes, maybe with supervisord.
# With this hot reloading you can make changes in the runpod container and see their effect immediately.
runpod.serverless.start({
    "handler": handler
})
