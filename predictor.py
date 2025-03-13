# app/predictor.py

from .models import ArtifactClassifier, DensityVotingClassifier, YoloCropper, YOLOVotingLesionDetector
from .models.density import BaseClassifier, EfficientNetClassifier, RegNetClassifier
from .models.detection import BaseDetector
from .preprocess import DICOMConverter
from .config import MamoConfig
from .utils import get_files_in_dir_recursively
from .constants import Usability
from .schemas import YOLOCropData, DensityResult
from .schemas.yolo_detection_results import YOLODetectionResult
from app.managers import ImagePath

from typing import List, Dict, Tuple
from loguru import logger

class Predictor:
    def __init__(self, cfg: MamoConfig) -> None:
        self.cfg = cfg
        self.converter = DICOMConverter()
        self.artifact_classifier = ArtifactClassifier(cfg)
        self.yolo_cropper = YoloCropper(cfg)
        self.density_models = self.initialize_density_models()
        self.density_classifier = DensityVotingClassifier(models=self.density_models)
        self.yolo_detector = self.initialize_yolo_voting_detector()

    def initialize_density_models(self) -> Dict[str, BaseClassifier]:
        models = {}
        for model_name, _ in self.cfg.density_model_local_paths.items():
            if "efficientnet" in model_name:
                models[model_name] = EfficientNetClassifier(
                    cfg=self.cfg,
                    model_name=model_name
                )
            elif "regnet" in model_name:
                models[model_name] = RegNetClassifier(
                    cfg=self.cfg
                )
            else:
                raise ValueError(f"Invalid model name: {model_name}")
        return models

    def initialize_yolo_voting_detector(self) -> YOLOVotingLesionDetector:
        """
        Initialize the YOLOVotingLesionDetector by loading models from local or bucket paths.
        """
        detectors = {}
        for detector_name, local_path in self.cfg.lesion_detection_model_local_paths.items():
            bucket_path = self.cfg.lesion_detection_model_bucket_paths.get(detector_name)

            if not local_path or not bucket_path:
                raise ValueError(f"Both local and bucket paths must be defined for detector: {detector_name}")

            detectors[detector_name] = BaseDetector(
                cfg=self.cfg,
                model_local_path=local_path,
                model_bucket_path=bucket_path
            )

        return YOLOVotingLesionDetector(
            detectors=detectors,
            iou_threshold=self.cfg.yolo_iou_threshold,
            benign_agreement_threshold=self.cfg.yolo_benign_threshold,
            malignant_agreement_threshold=self.cfg.yolo_malignant_threshold
        )


    def convert_to_png(self, dicom_paths: List[str]) -> Dict[str, str]:
        dicom_to_png = {}
        for dcm_path in dicom_paths:
            try:
                saved_png_path = self.converter.convert_to_png(
                    dicom_path=dcm_path, 
                    png_path=ImagePath.dcm_to_png(dcm_path)
                )
                if saved_png_path is not None:
                    dicom_to_png[dcm_path] = saved_png_path
            except Exception as e:
                logger.error(f"Error converting {dcm_path}: {str(e)}")

        return dicom_to_png

    def filter_by_artifact_classifier(self, dicom_to_png: Dict[str, str]) -> Dict[str, str]:
        classifier_results = self.artifact_classifier.predict(list(dicom_to_png.values()))

        filtered_dict = {}
        for dcm_path, png_path in dicom_to_png.items():
            class_name, prob = classifier_results[png_path]
            if class_name == Usability.NO_VISIBLE_TISSUE and prob > 0.5:
                logger.info(f"Filtered {dcm_path} because of no visible tissue")
                continue
            else:
                filtered_dict[dcm_path] = png_path

        return filtered_dict

    def crop_pngs(self, dicom_to_png: Dict[str, str]) -> Dict[str, YOLOCropData]:
        cropped_pngs = {}
        for dcm_path, png_path in dicom_to_png.items():
            crop_data = self.yolo_cropper.process_image(png_path)
            if crop_data is not None:
                cropped_pngs[dcm_path] = crop_data
            else:
                logger.info(f"Failed to crop {dcm_path}")

        return cropped_pngs

    def predict(self, dicom_folder_dir: str) -> Tuple[DensityResult, YOLODetectionResult]:
        """ 
        Predict the density and lesion detection results for the given DICOM folder.
        
        Args:
            dicom_folder_dir: Directory containing DICOM files
            
        Returns:
            Tuple of density result and YOLO detection result
        """
        
        dicom_paths = get_files_in_dir_recursively(dicom_folder_dir, ".dcm")
        
        logger.debug(f"Found {len(dicom_paths)} DICOM files in {dicom_folder_dir}")

        if self.cfg.tomo_to_mammo:
            dicom_paths = self.converter.convert_tomosynthesis(dicom_paths)

        # Step 1: DICOM to PNG Conversion        
        dicoms_to_png = self.convert_to_png(dicom_paths)

        # Step 2: Artifact Filtering
        dicoms_to_png = self.filter_by_artifact_classifier(dicoms_to_png)

        # Step 3: Cropping with YOLO
        cropped_dataset = self.crop_pngs(dicoms_to_png)
        # TODO add a function call to check to use the tomo data 
        
        # Step 4: Density Classification
        density_result, density_results_per_crop = self.density_classifier.predict(list(cropped_dataset.values()))

        # Step 5: YOLO Detection using density-cropped pairs
        # we pass density information as well, because in the future we might want to use it for hbreast scoring
        # if we will not, we can simplify it to cropped_dataset.values()
        yolo_detection_result = self.yolo_detector.predict(density_results_per_crop)

        return cropped_dataset, density_result, yolo_detection_result
