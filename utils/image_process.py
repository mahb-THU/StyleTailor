from PIL import Image


def image_process(image, target_width=768, target_height=1024):
    if image.mode != 'RGB':
        garment_image = image.convert('RGB')
    else:
        garment_image = image
    new_image = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    original_width, original_height = garment_image.size
    ratio_width = target_width / original_width
    ratio_height = target_height / original_height
    ratio = min(ratio_width, ratio_height)
    resized_width = int(original_width * ratio)
    resized_height = int(original_height * ratio)
    resized_garment_image = garment_image.resize((resized_width, resized_height), Image.LANCZOS)
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    new_image.paste(resized_garment_image, (x_offset, y_offset))
    return new_image
