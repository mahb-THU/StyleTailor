from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import ModelPlatformType, ModelType
import yaml
from jinja2 import Environment, StrictUndefined
from camel.configs import QwenConfig
from PIL import Image
import json
import re
from .parameter_dict import model_dict

def get_need2text(description, image_path):
    with open(f"code/utils/prompt_templates/prompts_text2image.yaml", "r") as f:
            config = yaml.safe_load(f)
    designer_model = ModelFactory.create(
                    model_platform=model_dict[0]["MODEL_PLATFORM"],
                    model_type=model_dict[0]["MODEL"],
                    model_config_dict=QwenConfig().as_dict(),
                )
    designer_sys_msg = config['system_prompt']
    designer_agent = ChatAgent(
            system_message=designer_sys_msg,
            model=designer_model,
            message_window_size=None,
        )

    jinja_env = Environment(undefined=StrictUndefined)
    designer_template = jinja_env.from_string(config["template"])
    jinja_args = {
                    'user_clothing_description':description
                }

    prompt = designer_template.render(**jinja_args)

    model_img = Image.open(image_path)
    msg = BaseMessage.make_user_message(
                role_name="User",
                content=prompt,
                image_list=[model_img],
            )
    response = designer_agent.step(msg)
    res = response.msgs[0].content
    match = re.search(r'```json\s*(.*?)\s*```|```\s*(.*?)\s*```', res, re.DOTALL)
    if match:
        json_candidate = match.group(1) if match.group(1) is not None else match.group(2) 
    else:
        json_candidate = res    
    data = json.loads(json_candidate)
    category_list = data["category"]
    prompt_dict = data["prompts"]
    designer_agent.reset()
    return category_list, prompt_dict

def get_addition_need2text(moe_order, description, negative_examples, image_path):
    with open(f"code/utils/prompt_templates/prompts_negative_moe.yaml", "r") as f:
            config = yaml.safe_load(f)
    designer_model = ModelFactory.create(
                    model_platform=model_dict[moe_order + 1]["MODEL_PLATFORM"],
                    model_type=model_dict[moe_order + 1]["MODEL"],
                    model_config_dict=QwenConfig().as_dict(),
                )
    designer_sys_msg = config['system_prompt']
    designer_agent = ChatAgent(
            system_message=designer_sys_msg,
            model=designer_model,
            message_window_size=None,
        )

    jinja_env = Environment(undefined=StrictUndefined)
    designer_template = jinja_env.from_string(config["template"])
    
    examples = ""
    for i, negative_example in enumerate(negative_examples):
        examples = examples + f"negative example {i + 1}:" + str(negative_example) + "\n"
        
    jinja_args = {
                    'user_clothing_description':description,
                    'negative_examples':examples
                }

    prompt = designer_template.render(**jinja_args)

    model_img = Image.open(image_path)
    msg = BaseMessage.make_user_message(
                role_name="User",
                content=prompt,
                image_list=[model_img],
            )
    response = designer_agent.step(msg)
    res = response.msgs[0].content
    match = re.search(r'```json\s*(.*?)\s*```|```\s*(.*?)\s*```', res, re.DOTALL)
    if match:
        json_candidate = match.group(1) if match.group(1) is not None else match.group(2) 
    else:
        json_candidate = res    
    data = json.loads(json_candidate)
    category_list = data["category"]
    prompt_dict = data["prompts"]
    designer_agent.reset()
    return category_list, prompt_dict