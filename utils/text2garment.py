from googleapiclient.discovery import build
from PIL import Image
from .metrics import VQAScore
import os
import shutil
from tqdm import tqdm
import random
import time
from .image_process import image_process
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import ModelPlatformType, ModelType
import yaml
import re
import json
from camel.configs import QwenConfig
from jinja2 import Environment, StrictUndefined
from .parameter_dict import garment_vqa_high_threshold, garment_vqa_low_threshold, max_text2garment_iterations

def get_addition_negative_prompt(text, garment_image):
    with open (f"code/utils/prompt_templates/prompts_negative_search.yaml", "r") as f:
        config = yaml.safe_load(f)
    
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
                    'user_clothing_description':text
                }

    prompt = critic_template.render(**jinja_args)
    
    msg = BaseMessage.make_user_message(
                role_name="User",
                content=prompt,
                image_list=[garment_image],
            )
    
    response = critic_agent.step(msg)
    res = response.msgs[0].content
    match = re.search(r'```json\s*(.*?)\s*```|```\s*(.*?)\s*```', res, re.DOTALL)
    if match:
        json_candidate = match.group(1) if match.group(1) is not None else match.group(2) 
    else:
        json_candidate = res
        
    data = json.loads(json_candidate)
    positives = data["positive_prompt"]
    negatives = data["negative_prompt"]
    
    critic_agent.reset()
    return positives, negatives
    
    

def make_queries(prompts, negative_prompts):
    negative_parts = []
    for keyword in negative_prompts:
        negative_parts.append(f'-"{keyword}"') 
    negative_query_string = " ".join(negative_parts)
    if prompts and negative_query_string:
        final_query = f"{prompts} {negative_query_string}"
    elif prompts:
        final_query = prompts
    elif negative_query_string:
        final_query = negative_query_string
    else:
        final_query = "" 
        
    negative_features_str = "、".join(negative_prompts)
    final_description = f"{prompts}. The image or objects within the image should not have the following characteristics:{negative_features_str}"
    
    return final_query, final_description

def get_text2garment(query, prefix_path, suffix_path):
    middle_paths = []
    result_links = []
    print(f"Searching Google Images for query: '{query}'...")
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        res = service.cse().list(
            q=query,
            cx=CX_ID,
            searchType='image'
        ).execute()
        if 'items' in res:
            print(f"Found {len(res['items'])} images. Starting download and resize...")
            for i, item in enumerate(tqdm(res['items'], desc="Downloading Images")):
                middle_path = f"{prefix_path}_{i}{suffix_path}"
                target_directory, file_name = os.path.split(middle_path)
                image_url = item.get('link')
                if image_url:
                    result_links.append(image_url)
                    delay = random.uniform(1, 5)
                    print(f"waiting for {delay:.2f} s ...")
                    time.sleep(delay)
                    downloading_command = f"wget -O {file_name} '{image_url}'"
                    current_path = os.path.join(os.getcwd(), file_name)
                    os.system(downloading_command)
                    moving_command = f"mv {current_path} {target_directory}"
                    os.system(moving_command)
                    img = Image.open(middle_path)
                    img = image_process(img)
                    img.resize((768, 1024))
                    img.save(middle_path)
                    middle_paths.append(middle_path)
                else:
                    print("Invalid Downloading Link")
        else:
            print(f"Images Not Found")

    except Exception as e:
        print(f"ERROR: {e}")
        
    return middle_paths, result_links

def produce_garment(prompts, negative_prompts, image_path, vqascore, threshold):
    # Split the image path into prefix and suffix
    prefix_path, suffix_path = os.path.splitext(image_path)
    google_query, vqa_prompt = make_queries(prompts=prompts, negative_prompts=negative_prompts)
    result_paths, result_links = get_text2garment(query=google_query, prefix_path=prefix_path, suffix_path=suffix_path)
    score_list = []
    print(f"Calculating VQA scores for {len(result_paths)} images...")
    for path in tqdm(result_paths, desc="Calculating VQA Scores"):
        score = vqascore.vqa_score(path, vqa_prompt)
        score_list.append(score)
        if score >= threshold:
            break  
    else:
        i = len(score_list) - 1
        
    # the ultimate result
    max_index = score_list.index(max(score_list)) if i == (len(result_paths) - 1) else i
    target_path = f"{prefix_path}_{max_index}{suffix_path}"
    shutil.copy(target_path, image_path)
    final_score = score_list[max_index]
    print("Final vqa_score", score_list[max_index])
    final_link = result_links[max_index]
    # Clean up all intermediate generated images
    for j in range(len(score_list)):
        os.remove(f"{prefix_path}_{j}{suffix_path}") 
        
    return final_link, final_score, vqa_prompt

def description_to_garment(prompts, negative_prompts, garment_paths, categories):
    """
    Convert a description to a garment image.
    """
    vqascore = VQAScore()
    print(prompts)
    garment_link = {}
    for category in tqdm(categories, desc="Processing Categories"):
        high_threshold = garment_vqa_high_threshold[category]
        low_threshold = garment_vqa_low_threshold[category]
        prompt = prompts[category]
        negative_prompt = negative_prompts[category]
        path = garment_paths[category]
        print(f"produce the {category} garment image ...")
        final_link, final_score, final_queries = produce_garment(prompts=prompt, 
                        negative_prompts=negative_prompt, 
                        image_path=path,
                        vqascore=vqascore,
                        threshold=high_threshold
                    )
        # negative feedback loop
        for _ in range(max_text2garment_iterations):
            if final_score > low_threshold:
                break
            else:
                garment_image = Image.open(path)
                new_positive_prompt, new_negative_prompt = get_addition_negative_prompt(text=final_queries, garment_image=garment_image)
                prompt += " " + ", ".join(new_positive_prompt)
                negative_prompt += new_negative_prompt
                    
                final_link, final_score, final_queries = produce_garment(prompts=prompt, 
                                                        negative_prompts=negative_prompt, 
                                                        image_path=path,
                                                        vqascore=vqascore,
                                                        threshold=high_threshold
                                            )
                
                
        garment_link[category] = final_link
    
    return garment_link
