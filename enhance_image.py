from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize
from models import MLP, normalized_pt, get_transform_to_params
from os.path import exists, join
from PIL import Image
import numpy as np
import random
import torch
import clip
import time
import os
import cloudinary
import cloudinary.uploader

app = FastAPI()

# Configure Cloudinary credentials
cloudinary.config( 
    cloud_name = "dj3p3xvrj",
    api_key = "838199179614134",
    api_secret = "FVlxqj5J5wYETH-a-wk6t5BEyDY"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
runif = np.random.uniform

# Load models once at startup
kwargs = {} if device == 'cuda' else {'map_location': torch.device('cpu')}
model_evalaesthetic = MLP(768)  
model_evalaesthetic.load_state_dict(torch.load("sac+logos+ava1-l14-linearMSE.pth", **kwargs))
model_evalaesthetic.to(device)
model_evalaesthetic.eval()
model_clip, preprocess = clip.load("ViT-L/14", device=device) 

def measure_aesthetic(pil_image, model_clip, model_evalaesthetic, flip=True):
    image = ToTensor()(pil_image)
    image = Normalize(
        (0.48145466, 0.4578275, 0.40821073), 
        (0.26862954, 0.26130258, 0.27577711)
    )(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        images = [image]
        if flip:
            images.append(torch.flip(image, dims=(2,)))
        aesthetic_values = []
        for img in images:
            image_features = model_clip.encode_image(img)
            im_emb_arr = normalized_pt(image_features)
            aesthetic_values.append(model_evalaesthetic(im_emb_arr.float()))
        aesthetic_value = torch.stack(aesthetic_values).mean(0)
    return round(aesthetic_value[0].item(), 4)

def get_random_transform(transforms_list, transform_to_params):
    t = random.choice(transforms_list)
    random_kwargs = transform_to_params[t]
    kwargs = {}
    for k, v in random_kwargs.items():
        if len(v) == 1:
            kwargs[k] = round(runif(*v[0]), 3)
        else:
            vx = v[0](*v[1])
            if v[0] != np.random.randint:
                vx = round(vx, 3)
            if k == 'kernel_size' and vx % 2 == 0:
                vx += 1
            kwargs[k] = vx
    return t, kwargs

def get_optimal_transform(image, it, transforms_list, transform_to_params, model_clip, model_evalaesthetic):
    t = transforms_list[it % len(transforms_list)]
    best_score = 0
    best_kwargs = {}
    param_keys = list(transform_to_params[t].keys())
    v2s = {}
    for key in param_keys:
        min_v = transform_to_params[t][key][-1][0]
        max_v = transform_to_params[t][key][-1][1]
        values_to_try = np.linspace(min_v, max_v, 10)
        if transform_to_params[t][key][0] == np.random.randint:
            values_to_try = set(int(np.floor(v)) for v in np.linspace(min_v, max_v - 1, 10))
        if key == 'kernel_size':
            values_to_try = [v for v in values_to_try if v % 2 != 0]
        for v in values_to_try:
            kwargs = {k: 1 for k in param_keys}
            kwargs[key] = v
            post_image = t(image, **kwargs)
            score = measure_aesthetic(post_image, model_clip, model_evalaesthetic)
            v2s[v] = score
            if score > best_score:
                best_score = score
                best_kwargs = kwargs
    return t, best_kwargs

def apply_node(image, node, dtree):
    if node == 0:
        return image
    elif isinstance(node, tuple):
        return node[0](image, **node[1])
    elif isinstance(node, int):
        nodes = dtree[node]
        for n in nodes:
            image = apply_node(image, n, dtree)
    return image

@app.post("/enhance-image")
async def enhance_image(file: UploadFile = File(...)):
    # Read image from upload
    image = Image.open(file.file).convert("RGB")
    base_pil_image = image
    base_pil_image_c = CenterCrop(224)(Resize(224)(base_pil_image))
    total_light_base = ToTensor()(base_pil_image).mean()

    # Initialize variables
    n_iter = 2000
    max_delay = 60  # Maximum runtime in seconds
    mode = 'soft'
    start_time = time.time()
    transform_to_params = get_transform_to_params(mode)
    transforms_list = list(transform_to_params.keys())

    # Decision tree initialization
    node_to_score = {}
    dtree = [[0]]
    base_score = measure_aesthetic(base_pil_image_c, model_clip, model_evalaesthetic)
    node_to_score[0] = base_score
    best_score = base_score
    best_node = 0
    it = 1
    last_improv_it = 0
    max_it_try_random = 500
    p_non_optimal = 0.8
    thres_bestscorelow = 0.2
    thres_basescorelow = 0.1
    ratio_light_thres = 0.85

    # Enhancement loop
    while time.time() - start_time < max_delay and it <= n_iter:
        # Decide previous node
        do_optimal = it < last_improv_it + len(transform_to_params)
        if it < max_it_try_random:
            prev_node = int(runif(0, max(1, len(dtree))))
        else:
            if runif(0, 1) < p_non_optimal and not do_optimal:
                prev_node = int(runif(0, len(dtree)))
                # Filter nodes
                if node_to_score[prev_node] < best_score - thres_bestscorelow or node_to_score[prev_node] < base_score - thres_basescorelow:
                    it += 1
                    continue
            else:
                prev_node = best_node

        # Apply previous node
        image = base_pil_image_c
        image = apply_node(image, prev_node, dtree)

        # Apply new transform
        if do_optimal:
            t, kwargs = get_optimal_transform(
                image, it, transforms_list, transform_to_params, model_clip, model_evalaesthetic
            )
        else:
            t, kwargs = get_random_transform(transforms_list, transform_to_params)
        image = t(image, **kwargs)

        # Measure aesthetic score
        score = measure_aesthetic(image, model_clip, model_evalaesthetic)

        # Adjust score based on lightness
        total_light_post = ToTensor()(image).mean()
        ratio_light = total_light_post / total_light_base
        if ratio_light < ratio_light_thres:
            score -= (1 - ratio_light) * 6

        # Update decision tree
        current_node = len(dtree)
        dtree.append([prev_node, (t, kwargs)])
        node_to_score[current_node] = score

        if score > best_score:
            best_score = score
            best_node = current_node
            last_improv_it = it

        it += 1

    # Apply best transformations
    enhanced_image = apply_node(base_pil_image, best_node, dtree)

    # Save enhanced image to a temporary file
    temp_image_path = "enhanced_image.jpg"
    enhanced_image.save(temp_image_path)

    # Upload to Cloudinary
    response = cloudinary.uploader.upload(temp_image_path)
    url = response.get('secure_url')

    # Clean up temporary file
    os.remove(temp_image_path)

    # Return the Cloudinary URL
    return {"url": url}

# To run the app, use the command: uvicorn main:app --reload