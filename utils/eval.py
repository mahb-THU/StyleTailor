from .metrics import VQAScore, FaceAnalyzer
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import ModelPlatformType, ModelType
import yaml
from jinja2 import Environment, StrictUndefined
from camel.configs import QwenConfig
from PIL import Image
import os
import json
import re

def final_eval(image_ref, image_result, description):
    img_ref = Image.open(image_ref)
    img_res = Image.open(image_result)
    
    # eval-1 : vqa_score
    final_vqa = final_vqa_score(image_ref=img_ref, image_result= image_result, description=description)
    # eval-2 : visual_quality
    final_clip = final_visual_quality(image_result=image_result)
    # eval-3 : face_id
    final_face = final_face_id(image_ref=image_ref, image_result=image_result)
    # eval-4 : vlm-commenter
    final_vlm = final_vlm_commenter(image_result=img_res)

    return final_vqa, final_clip, final_face, final_vlm

def final_vqa_score(image_ref, image_result, description):
    with open (f"code/utils/prompt_templates/prompts_metrics.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    commenter_model = ModelFactory.create(
                    model_platform=ModelPlatformType.QWEN,
                    model_type=ModelType.QWEN_VL_MAX,
                    model_config_dict=QwenConfig().as_dict(),
                )
    commenter_sys_msg = config['system_prompt']
    commenter_agent = ChatAgent(
            system_message=commenter_sys_msg,
            model=commenter_model,
            message_window_size=None,
        )
    jinja_env = Environment(undefined=StrictUndefined)
    designer_template = jinja_env.from_string(config["template"])
    prompt = designer_template.render()
    msg = BaseMessage.make_user_message(
                role_name="User",
                content=prompt,
                image_list=[image_ref],
            )
    response = commenter_agent.step(msg)
    res = response.msgs[0].content
    commenter_agent.reset()
    
    text = res + "A person with the characteristics above is now in the description below:" + description
    vqascore = VQAScore()
    score = vqascore.vqa_score(image_result, text)
    print("The final VQA Score is:", score)
    return score
    
def final_visual_quality(image_result):
    command = f"pyiqa qualiclip+ -t {image_result}"
    os.system(command)

def final_face_id(image_ref, image_result):
    face_analyzer = FaceAnalyzer()
    score = face_analyzer.compare_face_similarity(image_ref, image_result)
    print("The final Face ID Score is:", score)
    return score
    
def final_vlm_commenter(image_result):
    with open (f"code/utils/prompt_templates/prompts_eval.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    commenter_model = ModelFactory.create(
                    model_platform=ModelPlatformType.QWEN,
                    model_type=ModelType.QWEN_VL_MAX,
                    model_config_dict=QwenConfig().as_dict(),
                )
    commenter_sys_msg = config['system_prompt']
    commenter_agent = ChatAgent(
            system_message=commenter_sys_msg,
            model=commenter_model,
            message_window_size=None,
        )
    jinja_env = Environment(undefined=StrictUndefined)
    designer_template = jinja_env.from_string(config["template"])
    prompt = designer_template.render()
    msg = BaseMessage.make_user_message(
                role_name="User",
                content=prompt,
                image_list=[image_result],
            )
    response = commenter_agent.step(msg)
    res = response.msgs[0].content
    match = re.search(r'```json\s*(.*?)\s*```|```\s*(.*?)\s*```', res, re.DOTALL)
    if match:
        json_candidate = match.group(1) if match.group(1) is not None else match.group(2) 
    else:
        json_candidate = res    
    data = json.loads(json_candidate)
    design = int(data["design_rating"])
    fit = int(data["fit_rating"])
    coherence = int(data["coherence_rating"])
    mood = int(data["mood_rating"])
    final_score = (design + fit + coherence + mood) / 4
    
    
    
    commenter_agent.reset()
    print("VLM Score:", final_score)
    