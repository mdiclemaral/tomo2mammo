import os
import pydicom
import zipfile
import tempfile
from typing import List, Dict
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.transform import resize

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

def read_dicom_files(root_folder: str):
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
    print(f"Read {len(dataset)} DICOM files.")

    return dataset
def max_intensity_projection(images):
    """Computes the Maximum Intensity Projection (MIP) across all slices."""
    return np.max(images, axis=0)

def avg_intensity_projection(images):
    """Computes the Average Intensity Projection (AIP) across all slices."""
    return np.mean(images, axis=0)

def sum_intensity_projection(images):
    """Computes the Sum Intensity Projection (SIP) across all slices."""
    return np.sum(images, axis=0)

def save_image(image, filename):
    """Saves the processed image with correct contrast"""
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def normalize_image(image):
    """Normalizes image to range [0, 255] and converts to uint8"""
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
        arr[..., i] = s.pixel_array
    return arr

def process_dicom_patients(dicom_datasets, output_dir="tomo_to_mammo_output"):
    """Processes DICOM datasets for each patient, computes intensity projections, and saves images."""
    os.makedirs(output_dir, exist_ok=True)

    for patient_name, patient_dcm in dicom_datasets.items():
        reference_shape = None
        images_stack = []
        for img_dcm in patient_dcm:
            if hasattr(img_dcm, 'pixel_array'):
                image = img_dcm.pixel_array.astype(np.float32)
                
                if reference_shape is None:
                    reference_shape = image.shape  # Set reference shape to the first image shape
                
                # Resize image if it doesn't match the reference shape
                if image.shape != reference_shape:
                    image = resize(image, reference_shape, anti_aliasing=True, preserve_range=True)
                
                image = normalize_image(image)  # Normalize all images

                images_stack.append(image)
            else:
                print(f"Skipping: No pixel array found.")

        
        if images_stack:
            images_stack = np.array(images_stack)
            mip = max_intensity_projection(images_stack)
            aip = avg_intensity_projection(images_stack)
            sip = sum_intensity_projection(images_stack)

            # Save images
            save_image(mip, os.path.join(output_dir, f"{patient_name}_MIP.png"))
            save_image(aip, os.path.join(output_dir, f"{patient_name}_AIP.png"))
            save_image(sip, os.path.join(output_dir, f"{patient_name}_SIP.png"))
            
            print(f"Processed {patient_name}: MIP, AIP, SIP computed and saved.")
        else:
            print(f"Skipping {patient_name}: No valid images found.")



def save_dicom_as_png(dicom_datasets: Dict[str, pydicom.dataset.FileDataset],output_folder: str):
    """
    Saves DICOM images as PNG format.
    
    Args:
        dicom_datasets (List[pydicom.dataset.FileDataset]): List of DICOM datasets.
        output_folder (str): The folder to save PNG images.
    """
    os.makedirs(output_folder, exist_ok=True)
    for name, img_dcm_list in dicom_datasets.items():
        print(f"Processing {name}")
        i = 0
        for img_dcm in img_dcm_list:
            if hasattr(img_dcm, 'pixel_array'):
                image = img_dcm.pixel_array
                image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255  # Normalize
                image = 255 - image            
                image = image.astype(np.uint8)

                
                output_path = os.path.join(output_folder, f"{name}-{i}.png")
                plt.imsave(output_path, image, cmap='gray')
                print(f"Saved {output_path}")
                i += 1

# Example usage
root_folder = r"tomosynth\tomo_sm"
dicom_datasets = read_dicom_files(root_folder)
process_dicom_patients(dicom_datasets, "tomo_to_mammo_output")
# save_dicom_as_png(dicom_datasets, "sample_pngs")

print(f"Found {len(dicom_datasets)} DICOM files.")
