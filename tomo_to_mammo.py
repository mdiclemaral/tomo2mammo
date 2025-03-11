import os
import pydicom
from typing import List, Dict
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def find_dicom_files(root_folder: str) -> List[str]:
    """
    Recursively finds all DICOM (.dcm) files in the specified folder structure,
    including those inside ZIP archives.
    
    Args:
        root_folder (str): The root directory to start searching from.
    
    Returns:
        List[str]: A list of paths to DICOM files.
    """
    dicom_files = []

    
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            
            if filename.lower().endswith(".dcm"):
                dicom_files.append(file_path)

    print(f"Found {len(dicom_files)} DICOM files.")
    return dicom_files

def read_dicom_files(root_folder: str) -> Dict[str, List[pydicom.dataset.FileDataset]]:
    """
    Reads all DICOM files from the specified folder structure and returns them as pydicom datasets.
    
    Args:
        root_folder (str): The root directory to start searching from.
    
    Returns:
        List[pydicom.dataset.FileDataset]: A list of pydicom datasets.
    """
    dicom_files = find_dicom_files(root_folder)
    dataset = {}
    
    print(f"Reading {len(dicom_files)} DICOM files...")
    for i, dicom_file in enumerate(dicom_files):
        try:
            img = pydicom.dcmread(dicom_file)
            split_file_name = dicom_file.split("\\")
            patient_id = split_file_name[3]
            patient_study = split_file_name[-2]
            if (patient_id, patient_study) not in dataset:
                dataset[(patient_id, patient_study)] = []
            dataset[(patient_id, patient_study)].append(img)

        except Exception as e:
            print(f"Error reading {dicom_file}: {e}")
    return dataset

def save_image(image:np.array, filename:str) -> None:
    """Saves the processed image as a PNG file.
    Args:
        image (np.array): The image to save.
        filename (str): The filename to save the image as.
    
    Returns:
        None
    """ 
    plt.imsave(filename, image, cmap='gray')

def normalize_image(image: np.array) -> np.array:
    """Normalizes image to range [0, 255] and converts to uint8

    Args:
        image (np.array): The image to normalize.

    Returns:
        np.array: The normalized image
    
    """
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = 255 - image  # Invert image  
    return image.astype(np.uint8)

def get_numpy_array(axial_slices: List[pydicom.dataset.FileDataset]) -> np.array:
    """Converts a list of sorted DICOM slices into a 3D NumPy array.
    
    Args:
        axial_slices (List[pydicom.dataset.FileDataset]): List of DICOM slices.
        
    Returns:
        np.array: A 3D NumPy array of shape (height, width, num_slices).
    """
    arr = np.zeros((axial_slices[0].Rows, axial_slices[0].Columns, len(axial_slices)), dtype='int16')
    for i, s in enumerate(axial_slices):
        arr[..., i] = normalize_image(s.pixel_array.astype(np.float32))
    return arr

def max_intensity_projection(patient_dcm: List[pydicom.dataset.FileDataset]) -> np.array:
    """Computes the Maximum Intensity Projection (MIP) 

    Args:
        patient_dcm (List[pydicom.dataset.FileDataset]): List of DICOM slices for a patient.

    Returns:
        np.array: The MIP image.
    
    """

    arr_3d = get_numpy_array(patient_dcm)  # Convert to 3D array (H, W, num_slices)
    mip = np.max(arr_3d, axis=-1)  # Compute MIP across the last axis


    return mip

def avg_intensity_projection(patient_dcm: List[pydicom.dataset.FileDataset]) -> np.array:
    """Computes the Average Intensity Projection (AIP) 

    Args:
        patient_dcm (List[pydicom.dataset.FileDataset]): List of DICOM slices for a patient.

    Returns:
        np.array: The AIP image.
    """

    arr_3d = get_numpy_array(patient_dcm)  # Convert to 3D array (H, W, num_slices)
    aip = np.mean(arr_3d, axis=-1)  # Compute AIP across the last axis

    return aip

def process_dicom_patients(dicom_datasets: Dict[str, List[pydicom.dataset.FileDataset]]
                           , output_dir:str ="tomo_to_mammo_output") -> None:
    """Processes DICOM datasets for each patient, computes intensity projections, and saves images.
    
    Args:
        dicom_datasets (Dict[str, List[pydicom.dataset.FileDataset]]): A dictionary of DICOM datasets for each patient.
        output_dir (str): The output directory to save the images.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    for patient_name, patient_dcm in dicom_datasets.items():

        if len(patient_dcm) == 1:
            normalized_org_img = normalize_image(patient_dcm[0].pixel_array)
            save_image(normalized_org_img, os.path.join(output_dir, f"{patient_name}.png"))
            print(f"Saved original {patient_name}: Single image saved.")
            continue    

        mip = max_intensity_projection(patient_dcm)
        aip = avg_intensity_projection(patient_dcm)
        # Save images
        save_image(mip, os.path.join(output_dir, f"{patient_name}_MIP.png"))
        save_image(aip, os.path.join(output_dir, f"{patient_name}_AIP.png"))
        
        print(f"Processed {patient_name}: MIP, AIP, SIP computed and saved.")



# Example usage
root_folder = r"tomosynth\tomo"
dicom_datasets = read_dicom_files(root_folder)
process_dicom_patients(dicom_datasets, "all_tomo_to_mammo_output")