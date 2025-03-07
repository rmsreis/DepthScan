import numpy as np
from scipy.interpolate import interp1d

def enhance_profile_contrast(profiles, percentile_low=2, percentile_high=98):
    """
    Enhance the contrast of AFM profiles for better visualization.
    
    Args:
        profiles (numpy.ndarray): 2D array of profiles
        percentile_low (float): Lower percentile for contrast stretching
        percentile_high (float): Higher percentile for contrast stretching
        
    Returns:
        numpy.ndarray: Enhanced profiles with better contrast
    """
    # Calculate the percentile values for the entire dataset
    p_low = np.percentile(profiles, percentile_low)
    p_high = np.percentile(profiles, percentile_high)
    
    # Apply contrast stretching
    enhanced = np.clip((profiles - p_low) / (p_high - p_low), 0, 1)
    
    return enhanced

def interpolate_depth_profiles(depths, profiles, interp_factor=5, method='linear'):
    """
    Interpolate between AFM depth profiles to create a smoother 3D visualization.
    
    Args:
        depths (numpy.ndarray): Original depths where profiles were measured (in microns)
        profiles (numpy.ndarray): 2D array of profiles with shape (n_depths, profile_length)
        interp_factor (int): Factor by which to increase the number of depth slices
        method (str): Interpolation method ('linear', 'cubic', etc.)
    
    Returns:
        tuple: (interpolated_depths, interpolated_profiles) where:
            - interpolated_depths is a numpy array of new depths with more points
            - interpolated_profiles is a 2D numpy array of interpolated profiles
    """
    # Ensure inputs are numpy arrays
    depths = np.array(depths)
    profiles = np.array(profiles)
    
    # Calculate the number of interpolated depths
    n_orig_depths = len(depths)
    n_interp_depths = (n_orig_depths - 1) * interp_factor + 1
    
    # Create a new array of interpolated depths
    interp_depths = np.linspace(depths.min(), depths.max(), n_interp_depths)
    
    # Initialize the interpolated profiles array
    interp_profiles = np.zeros((n_interp_depths, profiles.shape[1]))
    
    # For each point in the profile, interpolate across depths
    for i in range(profiles.shape[1]):
        # Extract the profile values at this position across all depths
        profile_values = profiles[:, i]
        
        # Create the interpolation function
        if method == 'linear' and n_orig_depths >= 2:
            interp_func = interp1d(depths, profile_values, kind='linear')
        elif method == 'cubic' and n_orig_depths >= 4:
            interp_func = interp1d(depths, profile_values, kind='cubic')
        elif method == 'quadratic' and n_orig_depths >= 3:
            interp_func = interp1d(depths, profile_values, kind='quadratic')
        else:
            # Fallback to linear if not enough points or unsupported method
            interp_func = interp1d(depths, profile_values, kind='linear')
        
        # Apply the interpolation function to get new values
        interp_profiles[:, i] = interp_func(interp_depths)
    
    return interp_depths, interp_profiles

def create_depth_matrix(depths, profiles):
    """
    Create a 2D matrix representation of profiles across depths for heatmap visualization.
    
    Args:
        depths (numpy.ndarray): Array of depth values
        profiles (numpy.ndarray): 2D array of profiles with shape (n_depths, profile_length)
        
    Returns:
        tuple: (depth_matrix, x_coords, y_coords) where:
            - depth_matrix is a 2D numpy array for heatmap visualization
            - x_coords is a 1D array representing the spatial dimension
            - y_coords is a 1D array representing the depth dimension
    """
    # Create coordinate arrays
    x_coords = np.arange(profiles.shape[1])
    y_coords = depths
    
    # The profiles are already in the right format for a depth matrix
    # where each row is a profile at a particular depth
    depth_matrix = profiles
    
    return depth_matrix, x_coords, y_coords
