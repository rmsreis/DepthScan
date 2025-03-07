import os
import glob
import re
import numpy as np
from skimage import io

def parse_depth_from_filename(filename):
    """
    Extract depth information from the filename.
    Supports optional units like 'um', 'micron', or 'microns'.
    
    Args:
        filename (str): The filename containing depth information
        
    Returns:
        float: The parsed depth value in microns
    """
    match = re.search(r'(\d+)\s*(?:um|micron(?:s)?)?', filename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Could not parse depth from filename: {filename}")

def prepare_image(image, crop_right=50):
    """
    Crop out the embedded colorbar from the right side of the image.
    
    Args:
        image (numpy.ndarray): The input image
        crop_right (int): Number of pixels to crop from the right
        
    Returns:
        numpy.ndarray: The cropped image
    """
    if image.shape[1] <= crop_right:
        raise ValueError("crop_right is larger than the image width")
    return image[:, :image.shape[1] - crop_right]

def load_depth_profiles(data_dir, file_pattern="*.jpg", crop_right=50):
    """
    Loads AFM profiles from images in the data directory.
    
    Args:
        data_dir (str): Directory containing AFM profile images
        file_pattern (str): Pattern to match image files
        crop_right (int): Number of pixels to crop from the right of each image
        
    Returns:
        tuple: (sorted_depths, profiles)
            - sorted_depths: list of depths (in microns) sorted in ascending order
            - profiles: 2D numpy array (rows: depth slices, columns: spatial coordinate)
    """
    # Get all image files in the directory matching the pattern
    image_files = glob.glob(os.path.join(data_dir, file_pattern))
    
    if not image_files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' found in '{data_dir}'")
    
    # Parse depths and load images
    depths = []
    profiles_list = []
    
    for img_file in image_files:
        try:
            # Load the image
            img = io.imread(img_file)
            
            # Convert to grayscale if the image is RGB
            if len(img.shape) == 3:
                img = np.mean(img, axis=2).astype(np.uint8)
            
            # Crop the image to remove the embedded colorbar
            cropped_img = prepare_image(img, crop_right)
            
            # Extract the central row as the 1D profile
            central_row = cropped_img[cropped_img.shape[0] // 2, :]
            
            # Parse the depth from the filename
            depth = parse_depth_from_filename(os.path.basename(img_file))
            
            depths.append(depth)
            profiles_list.append(central_row)
            
            print(f"Loaded profile at depth {depth} µm from {os.path.basename(img_file)}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    # Sort profiles by depth
    depth_indices = np.argsort(depths)
    sorted_depths = np.array([depths[i] for i in depth_indices])
    sorted_profiles = np.array([profiles_list[i] for i in depth_indices])
    
    # Normalize profiles to a consistent range for better visualization
    sorted_profiles = sorted_profiles.astype(float)
    for i in range(len(sorted_profiles)):
        profile = sorted_profiles[i]
        if profile.max() != profile.min():
            sorted_profiles[i] = (profile - profile.min()) / (profile.max() - profile.min())
    
    return sorted_depths, sorted_profiles
