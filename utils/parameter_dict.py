from camel.types import ModelPlatformType, ModelType

category_dict_utils = [
    'upper_body', 
    'lower_body',
    'dresses',
    'shoes',
    'hat',
    'glasses',
    'belt',
    'scarf'
]

max_text2garment_iterations = 10
garment_vqa_high_threshold = {
    'upper_body' : 0.9,
    'lower_body' : 0.9,
    'dresses' : 0.9,
    'shoes' : 0.9,
    'hat' : 0.9,
    'glasses' : 0.9,
    'belt' : 0.9,
    'scarf' : 0.9
}

garment_vqa_low_threshold = {
    'upper_body' : 0.7,
    'lower_body' : 0.7,
    'dresses' : 0.7,
    'shoes' : 0.6,
    'hat' : 0.6,
    'glasses' : 0.6,
    'belt' : 0.6,
    'scarf' : 0.6
}

max_flux_vton_iterations = 10
vton_clip_score_high_threshold = {
    'upper_body' : 0.9,
    'lower_body' : 0.9,
    'dresses' : 0.9,
    'shoes' : 0.9,
    'hat' : 0.9,
    'glasses' : 0.9,
    'belt' : 0.9,
    'scarf' : 0.9
}

vton_clip_score_low_threshold = {
    'upper_body' : 0.7, #0.9,
    'lower_body' : 0.7, #0.9,
    'dresses' : 0.7, #0.9,
    'shoes' : 0.5, #0.9,
    'hat' : 0.5, #0.9,
    'glasses' : 0.6, #0.9,
    'belt' : 0.6, #0.9,
    'scarf' : 0.6 #0.9
}



q_moe_iterations = 10
q_moe_threshold = 0.65 #0.8

model_dict = [
    {
        'MODEL_PLATFORM':ModelPlatformType.OPENROUTER,
        'MODEL':"anthropic/claude-sonnet-4",
    },
    {
        'MODEL_PLATFORM':ModelPlatformType.OPENROUTER,
        'MODEL':"google/gemini-2.5-pro",
    },
    {
        'MODEL_PLATFORM':ModelPlatformType.OPENROUTER,
        'MODEL':"meta-llama/llama-4-maverick",
    },
    {
        'MODEL_PLATFORM':ModelPlatformType.QWEN,
        'MODEL':ModelType.QWEN_VL_MAX,
    },
    
]


    
