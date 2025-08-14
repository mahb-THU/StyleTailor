from .metrics import VQAScore
from .parameter_dict import q_moe_iterations, q_moe_threshold
from PIL import Image
import numpy as np
from .text2garment import description_to_garment
from .need2text import get_addition_need2text

def get_moe_score(category_list, garment_path_dict, user_description, vqascore):
    score_list = []
    for category in category_list:
        # garment_image = Image.open()
        text = f"Here is the {category} image which is fit for the user's need:" + user_description
        score = vqascore.vqa_score(garment_path_dict[category], text)
        score_list.append(score)
        
    score_list = np.array(score_list)
    log_score = np.log(score_list)
    mean_log_score = np.mean(log_score)
    final_score = np.exp(mean_log_score)
    
    return final_score

def q_moe_addition(garment_path_dict, user_description, category_list, model_image_path, negative_prompt_dict, fail_prompt_dict, init_link):   
    negative_examples = []
    fail_example = fail_prompt_dict
    vqascore = VQAScore()
    garment_link = init_link
    for i in range(q_moe_iterations):
        final_score = get_moe_score(category_list=category_list, garment_path_dict=garment_path_dict, user_description=user_description, vqascore=vqascore)
        print("Q_MOE: Final score = " + str(final_score))
        if final_score > q_moe_threshold:
            break
        else:
            negative_examples.append(fail_example)
            
            category_list, prompt_dict = get_addition_need2text(i, user_description, negative_examples, model_image_path)
            garment_link = description_to_garment(
                    prompts = prompt_dict,
                    negative_prompts = negative_prompt_dict,
                    garment_paths = garment_path_dict,
                    categories=category_list
                )
            fail_example = prompt_dict
    
    return garment_link