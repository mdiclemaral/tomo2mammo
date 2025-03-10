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

def extract_zip(zip_path: str, extract_to: str):
    """
    Extracts a ZIP file to a temporary directory.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

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
    temp_dirs = []  # Track temporary extraction directories
    
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            
            if filename.lower().endswith(".dcm"):
                dicom_files.append(file_path)
            elif filename.lower().endswith(".zip"):
                # Create a temporary directory to extract ZIP contents
                temp_dir = tempfile.mkdtemp()
                temp_dirs.append(temp_dir)
                
                try:
                    extract_zip(file_path, temp_dir)
                    dicom_files.extend(find_dicom_files(temp_dir))
                except Exception as e:
                    print(f"Error extracting {file_path}: {e}")
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
            patient_id =dicom_file.split("\\")[3]
            if patient_id not in dataset:
                dataset[patient_id] = []
            dataset[patient_id].append(img)

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
    """Saves the processed image as a PNG file."""
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_dicom_patients(dicom_datasets, output_dir="tomo_to_mammo_output"):
    """Processes DICOM datasets for each patient, computes intensity projections, and saves images."""
    os.makedirs(output_dir, exist_ok=True)
    print(dicom_datasets.keys())
    for patient_name, patient_dcm in dicom_datasets.items():
        reference_shape = None
        print(f"Processing {patient_name}")
        images_stack = []
        for img_dcm in patient_dcm:
            if hasattr(img_dcm, 'pixel_array'):
                image = img_dcm.pixel_array
                
                if reference_shape is None:
                    reference_shape = image.shape  # Set reference shape to the first image shape
                
                # Resize image if it doesn't match the reference shape
                if image.shape != reference_shape:
                    image = resize(image, reference_shape, anti_aliasing=True, preserve_range=True)
                    image = image.astype(np.uint16)  # Convert back to uint16 if needed
                
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
    for name, img_dcm in dicom_datasets.items():
        print(f"Processing {name}")

        if hasattr(img_dcm, 'pixel_array'):
            image = img_dcm.pixel_array
            image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255  # Normalize
            image = image.astype(np.uint8)
            
            output_path = os.path.join(output_folder, f"{name}.png")
            plt.imsave(output_path, image, cmap='gray')
            print(f"Saved {output_path}")

# Example usage
root_folder = r"tomosynth\tomo_sm"
dicom_datasets = read_dicom_files(root_folder)
process_dicom_patients(dicom_datasets, "tomo_to_mammo_output")
# save_dicom_as_png(dicom_datasets, "sample_pngs")

print(f"Found {len(dicom_datasets)} DICOM files.")
