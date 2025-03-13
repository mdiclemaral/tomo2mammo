# app/models/yolo_cropper.py
from ..managers import AWSManager
from ultralytics import YOLO
import torch
from loguru import logger
import cv2
from typing import Any, Dict, Optional, List
from app.schemas import YOLOCropData
from app.managers import ImagePath


class YoloCropper(AWSManager):
    def __init__(self, cfg) -> None:
        AWSManager.__init__(self, cfg.model_bucket, cfg.yolo_cropper_bucket_path, 
                         cfg.yolo_cropper_local_path, 
                         cfg.aws_access_key, cfg.aws_secret_key) 
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(self.model_local_path)
        self.class_names = {0: "L-MLO", 1: "R-MLO", 2: "L-CC", 3: "R-CC"}
        
    def load_model(self, model_path: str):
        return YOLO(model_path).to(self.device)
    
    def load_image(self, img_path: str) -> Optional[Any]:
        """Load an image and convert to grayscale if necessary."""
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error(f"Failed to load image: {img_path}")
            return None

        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def process_image(self, img_path: str) -> Optional[YOLOCropData]:
        """Run prediction on a single image and save results."""
        imgsz = 928
        best_max_conf = 0
        best_max_conf_box = None
        current_max_conf_box = None

        while imgsz <= 1280:
            results = self.model.predict(img_path, conf=0.00001, device=self.model.device, imgsz=imgsz, optimize=True, verbose=False)
            boxes = results[0].boxes

            if boxes:
                current_max_conf_box = max(boxes, key=lambda box: box.conf.item())
                current_max_conf = current_max_conf_box.conf.item()

                if current_max_conf > best_max_conf:
                    best_max_conf = current_max_conf
                    best_max_conf_box = current_max_conf_box

                if best_max_conf >= 0.15:
                    break

            imgsz += 32

        if best_max_conf_box:
            img = self.load_image(img_path)
            crop_path = ImagePath.raw_to_crop(img_path)
            return self.save_result(box=best_max_conf_box, img=img, imgsz=imgsz, crop_path=crop_path)
        else:
            logger.info(f"No boxes found for {img_path}")

    def save_result(self, box: Any, img: Any, imgsz: int, crop_path: str) -> None:
        """Save cropped result and create a JSON file."""
        
        bbox = box.xyxy[0].cpu().numpy()
        cropped_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        cv2.imwrite(crop_path, cropped_img)


        crop_data = YOLOCropData(
                coordinates=(int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])),
                class_score=round(float(box.conf.item()), 6),
                series_description=self.class_names[int(box.cls)],
                img_size_used=imgsz,
                path=crop_path
            )

        return crop_data
    
    def predict(self, image_paths: List[str]) -> Dict[str, YOLOCropData]:
        results = {}
        for image_path in image_paths:    
            try:
                res = self.process_image(image_path)
                if res:
                    results[image_path] = res
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                
        return results
