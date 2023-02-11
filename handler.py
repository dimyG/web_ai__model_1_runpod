import runpod
import os
import io
import base64
from ml import load_pipeline, img_from_prompt
# from main import generate_img

# We read the defined cache dir from the env file, and then we use the cache_dir parameter to use it.
# Using the cache_dir parameter could be avoided if we set the HUGGINGFACE_HUB_CACHE environment variable on the shell.
# But it doesn't work just by adding it in the env file, so we read it and then use it as the cache_dir parameter.
huggingface_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")

model1 = "CompVis/stable-diffusion-v1-4"
model2 = "stabilityai/stable-diffusion-2"
# Runpod advice: load your model(s) into vram here
pipe_initial = load_pipeline(model2, cache_dir=huggingface_hub_cache)


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
    seed = int(event['input']['seed'])
    height = int(event['input']['height'])
    width = int(event['input']['width'])
    guidance_scale = float(event['input']['guidance_scale'])
    num_inference_steps = int(event['input']['num_inference_steps'])

    # Runpod requirement: If the image is larger than 2MB, store it in a blob store and return an url to it.
    image = img_from_prompt(prompt=prompt, pipe=pipe_initial, seed=seed, height=height, width=width,
                            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
    memory_stream = io.BytesIO()
    image.save(memory_stream, format='PNG')  # write the image to memory instead of the disk, and return it from there
    memory_stream.seek(0)
    # Convert the contents of the memory stream to a base64 encoded string
    encoded_string = base64.b64encode(memory_stream.getvalue()).decode('utf-8')
    return encoded_string


runpod.serverless.start({
    "handler": handler
})
