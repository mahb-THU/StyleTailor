import torch
import os
from diffusers import FluxKontextPipeline
from PIL import Image
from .human_mask.ootd_mask import get_mask_image
from .parameter_dict import category_dict_utils, vton_clip_score_high_threshold, vton_clip_score_low_threshold, max_flux_vton_iterations
from .metrics import ClipScore, HumanDetection
import gc
import yaml
from jinja2 import Environment, StrictUndefined
from .image_process import image_process
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import ModelPlatformType, ModelType
import yaml
import re
import json
from camel.configs import QwenConfig
import io
import shutil

# Force PNG format
def fix_image_format(image, format='PNG'):
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return Image.open(buffer)

def get_addition_negative_flux_prompt(image_ref, prompt_ref, image_result):
    with open (f"code/utils/prompt_templates/prompts_negative_flux.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    image_ref = fix_image_format(image_ref)
    image_result = fix_image_format(image_result)

    critic_model = ModelFactory.create(
                    model_platform=ModelPlatformType.QWEN,
                    model_type=ModelType.QWEN_VL_MAX,
                    model_config_dict=QwenConfig().as_dict(),
                )
    critic_sys_msg = config['system_prompt']
    critic_agent = ChatAgent(
            system_message=critic_sys_msg,
            model=critic_model,
            message_window_size=None,
        )
    
    jinja_env = Environment(undefined=StrictUndefined)
    critic_template = jinja_env.from_string(config["template"])
    jinja_args = {
                    'user_clothing_description':prompt_ref
                }

    prompt = critic_template.render(**jinja_args)
    
    msg = BaseMessage.make_user_message(
                role_name="User",
                content=prompt,
                image_list=[image_ref, image_result],
            )
    
    response = critic_agent.step(msg)
    res = response.msgs[0].content
    match = re.search(r'```json\s*(.*?)\s*```|```\s*(.*?)\s*```', res, re.DOTALL)
    if match:
        json_candidate = match.group(1) if match.group(1) is not None else match.group(2) 
    else:
        json_candidate = res
        
    data = json.loads(json_candidate)
    positive_str = data["positive_prompt"]
    negative_str = data["negative_prompt"]
    
    critic_agent.reset()
    return positive_str, negative_str

def contact_picture(image_left, image_right):
    image1 = image_left
    image2 = image_right
  
    height = max(image1.height, image2.height)
    image1 = image1.resize((int(image1.width * height / image1.height), height))
    image2 = image2.resize((int(image2.width * height / image2.height), height))

    new_width = image1.width + image2.width
    new_image = Image.new('RGB', (new_width, height))

    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))

    return new_image

def edit_vton_once(input_image, prompt, negative_prompt, pipe, guidance_scale=2.5, width=768, height=1024, num_images_per_prompt=3):
    images = pipe(
      image=input_image,
      prompt=prompt,
      negative_prompt=negative_prompt,
      guidance_scale=guidance_scale,
      width=width,
      height=height,
      num_images_per_prompt=num_images_per_prompt,
    ).images

    results = []
    for image in images:
        image_new = image.resize((768, 1024))
        results.append(image_new)
    
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return results

def pick_vton_once(model_image, garment_image, category, prompts, negative_prompt, clipscore, pipe, guidance_scale=2.5, width=768, height=1024, num_images_per_prompt=3, threshold=0.8):
    contact_image = contact_picture(model_image, garment_image)
    images = edit_vton_once(input_image=contact_image,
                            prompt=prompts,
                            negative_prompt=negative_prompt,
                            pipe=pipe,
                            guidance_scale=guidance_scale,
                            width=width,
                            height=height,
                            num_images_per_prompt=num_images_per_prompt
            )
    score_list = []
    for i, image in enumerate(images):
        masked_image = get_mask_image(model_image=image,
                                    gpu_id=0, 
                                    category=category
                    )
        score = clipscore.get_similarity_score(masked_image, garment_image)
        score_list.append(score)
        print("garment similarity score", score)
        if score >= threshold:
            break
    else:
        i = len(score_list) - 1
        
    max_index = score_list.index(max(score_list)) if i == len(images) - 1 else i
    final_score = score_list[max_index]
    result_image = images[max_index]
    
    return result_image, final_score
    
def run_vton(model_image_path, garment_paths, categories, prompts, output_path, guidance_scale=2.5, width=768, height=1024, num_images_per_prompt=3):
    model_image = Image.open(model_image_path)
    model_image = image_process(model_image)
    clipscore = ClipScore()
    humandetection = HumanDetection()
    pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    negative_prompt = ""
    for i, category in enumerate(categories):
        
        category_index = category_dict_utils.index(category)
        garment_path = garment_paths[category]
        garment_image = Image.open(garment_path)
        high_threshold = vton_clip_score_high_threshold[category]
        low_threshold = vton_clip_score_low_threshold[category]

        # Prompt generation depending on if the garment image has human models in it.
        if humandetection.detection(garment_image) == True:
            # Parse base prompts
            with open('code/utils/prompt_templates/prompts_flux_true.yaml', 'r', encoding='utf-8') as file:
                template_config = yaml.safe_load(file)
        else:
            with open('code/utils/prompt_templates/prompts_flux_false.yaml', 'r', encoding='utf-8') as file:
                template_config = yaml.safe_load(file)

        env = Environment(undefined=StrictUndefined)
        template_str = template_config['system_prompt']
        template = env.from_string(template_str)
        gender = prompts["gender"]
        description = prompts[category+"_short"]
        rendered_prompt = template.render(gender=gender, description=description)

        # Apply vton
        try_on_one_image, final_score = pick_vton_once(model_image=model_image,
                                        garment_image=garment_image,
                                        category=category_index,
                                        prompts=rendered_prompt,
                                        negative_prompt=negative_prompt,
                                        clipscore=clipscore,
                                        pipe=pipe,
                                        guidance_scale=guidance_scale,
                                        width=width, 
                                        height=height, 
                                        num_images_per_prompt=num_images_per_prompt, 
                                        threshold=high_threshold
                            )
        
        for _ in range(max_flux_vton_iterations):
            if final_score > low_threshold:
                break
            else:
                new_positive_prompt, new_negative_prompt = get_addition_negative_flux_prompt(image_ref=model_image, prompt_ref=rendered_prompt, image_result=try_on_one_image)
                negative_prompt = negative_prompt + ", ".join(new_negative_prompt)
                positive_prompt = rendered_prompt + ", ".join(new_positive_prompt)
                try_on_one_image, final_score = pick_vton_once(model_image=model_image,
                                        garment_image=garment_image,
                                        category=category_index,
                                        prompts=positive_prompt,
                                        negative_prompt=negative_prompt,
                                        clipscore=clipscore,
                                        pipe=pipe,
                                        guidance_scale=guidance_scale,
                                        width=width, 
                                        height=height, 
                                        num_images_per_prompt=num_images_per_prompt, 
                                        threshold=high_threshold
                            ) 

                
        model_image = try_on_one_image
        name, _ = os.path.splitext(os.path.basename(model_image_path))
        final_path = os.path.join(output_path, f"{name}_result_{i}_image.png")
        model_image.save(final_path)
    
    total = len(categories)
    name, _ = os.path.splitext(os.path.basename(model_image_path))
    target_path = os.path.join(output_path, f"{name}_result_{total - 1}_image.png")
    final_path = os.path.join(output_path, f"{name}_result_image.png")
    shutil.copy(target_path, final_path)
    
    for j in range(total):
        os.remove(os.path.join(output_path, f"{name}_result_{j}_image.png"))
  