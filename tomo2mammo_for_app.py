import os
import pydicom
import numpy as np
from typing import Dict, List
from loguru import logger

class TomosynthesisConverter:
    """Handles the conversion of tomosynthesis DICOMs into mammogram-equivalent images."""

    def __init__(self, dicom_paths: List[str] = None, delete_tomo_dcm: bool = True):
        """Initializes the converter with DICOM file paths and processing options."""
        self.dicom_paths = dicom_paths if dicom_paths is not None else []
        self.delete_tomo_dcm = delete_tomo_dcm
        logger.info("TomosynthesisConverter initialized.")

    def update_dcm_image(self, ds, new_pixel_array: np.array, output_dcm_path: str) -> None:   
        """Updates the DICOM image with a new pixel array and saves the updated DICOM file."""
        ds.PixelData = new_pixel_array.tobytes()
        ds.Rows, ds.Columns = new_pixel_array.shape
        ds.Modality = "S-MAMMO"
        ds.save_as(output_dcm_path)

    def normalize_image(self, image: np.array) -> np.array:
        """Normalizes image to range [0, 255] and converts to uint8."""
        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        image = 255 - image  
        return image.astype(np.uint8)

    def get_numpy_array(self, axial_slices: List[pydicom.dataset.FileDataset]) -> np.array:
        """Converts a list of sorted DICOM slices into a 3D NumPy array."""
        arr = np.zeros((axial_slices[0].Rows, axial_slices[0].Columns, len(axial_slices)), dtype='int16')
        for i, s in enumerate(axial_slices):
            arr[..., i] = self.normalize_image(s.pixel_array.astype(np.float32))
        return arr

    def max_intensity_projection(self, patient_dcm: List[pydicom.dataset.FileDataset]) -> np.array:
        """Computes the Maximum Intensity Projection (MIP)."""
        arr_3d = self.get_numpy_array(patient_dcm)
        return np.max(arr_3d, axis=-1)   

    def convert_tomosynthesis(self) -> List[str]: 
        """Processes tomosynthesis DICOMs and saves MIP images."""
        tomo_datasets = {}
        tomo_dcm_paths = {}

        for dcm_path in self.dicom_paths:
            try:
                dicom_data = pydicom.dcmread(dcm_path)
                series_UID = dicom_data.SeriesInstanceUID
                if series_UID not in tomo_datasets:
                    tomo_datasets[series_UID] = []
                tomo_datasets[series_UID].append(dicom_data)
                tomo_dcm_paths[series_UID] = dcm_path
            except Exception as e:
                logger.error(f"Error reading {dcm_path}: {str(e)}")

        for series_UID, patient_dcm in tomo_datasets.items():
            if len(patient_dcm) >= 10: # rearreange if needed (to )
                patient_tomo_dcm_paths = tomo_dcm_paths[series_UID]
                mip = self.max_intensity_projection(patient_dcm)
                output_dcm_path = patient_tomo_dcm_paths[0].replace(".dcm", "_smammo.dcm") # Change it to unique dicom path (use pydicom)
                self.update_dcm_image(patient_dcm, mip, output_dcm_path)
                
                if self.delete_tomo_dcm: # Change if we dor't want to delete the tomosynth imgs 
                    os.remove(patient_tomo_dcm_paths) if os.path.exists(patient_tomo_dcm_paths) else logger.warning(f"File not found: {patient_tomo_dcm_paths}")
                    self.dicom_paths.remove(patient_tomo_dcm_paths)
                self.dicom_paths.append(output_dcm_path)

        return self.dicom_paths
