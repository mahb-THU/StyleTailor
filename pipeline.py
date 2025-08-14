# -*- coding: utf-8 -*-
import os
import argparse
from .utils.flux_vton import run_vton
from .utils.text2garment import description_to_garment
from .utils.need2text import get_need2text
from .utils.parameter_dict import category_dict_utils
from .utils.q_moe import q_moe_addition
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StyleTailor Agent Pipeline")
    parser.add_argument("--garment_image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_image_path", type=str, required=True)
    parser.add_argument("--description", type=str, required=True)
    args = parser.parse_args()

    model_image_path = args.model_image_path
    name, _ = os.path.splitext(os.path.basename(model_image_path))
    user_description = args.description
    garment_image_path = args.garment_image_path
    output_path = args.output_path
    checkpoint_path = args.checkpoint_path
    
    os.makedirs(garment_image_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    garment_path_dict = {}
    for category in category_dict_utils:
        garment_path_dict[category] = os.path.join(garment_image_path, f"{name}_{category}_image.jpg")
    
    base_negative_prompt = ["worn by model", "low resolution", "incomplete", "multiple objects", "no white border", "partial clothing", "not front view for clothing", "not flat lay"]
    negative_prompt_dict = {}
    for category in category_dict_utils:
        negative_prompt_dict[category] = base_negative_prompt
    
    # Simulate pipeline execution
    print("Starting StyleTailor Agent Pipeline...")
     
    # 1. run q-moe
    # 1.1 user's need to garment description
    category_list, prompt_dict = get_need2text(user_description, model_image_path)
        
    # 1.2 description to garment image (generation and evaluation)
    garment_link = description_to_garment(
                    prompts = prompt_dict,
                    negative_prompts = negative_prompt_dict,
                    garment_paths = garment_path_dict,
                    categories=category_list
                )
    
    # 1.3 addition moe system
    garment_link = q_moe_addition(garment_path_dict=garment_path_dict,
                                user_description=user_description,
                                category_list=category_list,
                                model_image_path=model_image_path,
                                negative_prompt_dict=negative_prompt_dict,
                                fail_prompt_dict=prompt_dict,
                                init_link=garment_link
                    )              
    # 2. use the virtual try-on system
    run_vton(
        model_image_path=model_image_path,
        garment_paths=garment_path_dict,
        prompts=prompt_dict,
        categories=category_list,
        output_path=output_path,
        guidance_scale=2.5,
        width=768,
        height=1024,
        num_images_per_prompt=3
    )
    
    print("garment links", garment_link)
    print("Pipeline executed successfully with the following configuration:")
