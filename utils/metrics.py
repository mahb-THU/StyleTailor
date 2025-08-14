import t2v_metrics
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2
from insightface.app import FaceAnalysis
from ultralytics import YOLO
import numpy as np

# VQAScore
class VQAScore:
    def __init__(self):
        self.clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl', cache_dir="hf_cache")
    
    def vqa_score(self, images, texts):
        score = self.clip_flant5_score(images=[images], texts=[texts])
        return score.item()
    
# Visual Quality Score
class ClipScore:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)

    def get_similarity_score(self, image1, image2):
        image1 = image1.convert('RGB')
        image2 = image2.convert('RGB')
        inputs = self.processor(images=[image1, image2], return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        similarity_score_torch = torch.nn.functional.cosine_similarity(image_features[0], image_features[1], dim=0).item()
        return similarity_score_torch
    
# Face Comparison Score
class FaceAnalyzer:
    def __init__(self):
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def cosine_distance(self, embedding1, embedding2):
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        cosine_similarity = dot_product / (norm1 * norm2)
        
        cosine_distance = 1 - cosine_similarity
        
        return cosine_distance

    def cosine_similarity(self, embedding1, embedding2):
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        return dot_product / (norm1 * norm2)
        
    def compare_face_similarity(self, image_path1, image_path2):
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)
        # Detect and extract facial features
        faces1 = self.app.get(img1)
        faces2 = self.app.get(img2)
        # If multiple faces are detected, usually the first face (main face) is taken for comparison
        face_embedding1 = faces1[0].embedding
        face_embedding2 = faces2[0].embedding
        # cosine similarity
        similarity_score = self.cosine_similarity(face_embedding1, face_embedding2)
        return similarity_score

class HumanDetection:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
    
    def detection(self, image, confidence_threshold=0.5):
        results = self.model(source=image, conf=confidence_threshold)
        has_human = False
        for result in results:
            for box in result.boxes:
                class_name = self.model.names[int(box.cls)]
                confidence = float(box.conf)
                if class_name == 'person' and confidence >= confidence_threshold:
                    has_human = True
        return has_human