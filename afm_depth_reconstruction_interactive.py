import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from skimage import io, exposure
import ipywidgets as widgets
from ipywidgets import interactive, Layout, Button, HBox, VBox, Tab, Output, Dropdown, FloatSlider, Checkbox
from IPython.display import display, HTML, clear_output
from scipy.interpolate import RegularGridInterpolator, interp1d, griddata
import io as python_io
import base64
import time
import colorsys

def create_custom_colormap(name='afm_colormap', n_colors=256):
    """
    Create a custom colormap optimized for AFM data visualization.
    
    Args:
        name (str): Name for the custom colormap.
        n_colors (int): Number of colors in the colormap.
        
    Returns:
        matplotlib.colors.LinearSegmentedColormap: Custom colormap object.
    """
    # Create a visually striking colormap that highlights material differences
    # Using a palette that transitions from dark blues to teals to yellows to reds
    colors_list = []
    
    # Add deep blue to teal transition (first 40%)
    for i in range(int(n_colors * 0.4)):
        t = i / (n_colors * 0.4)
        r = 0.0 * (1-t) + 0.0 * t  # Staying at 0
        g = 0.0 * (1-t) + 0.8 * t  # 0 to 0.8
        b = 0.5 * (1-t) + 1.0 * t  # 0.5 to 1.0
        colors_list.append((r, g, b))
    
    # Add teal to yellow transition (next 30%)
    for i in range(int(n_colors * 0.3)):
        t = i / (n_colors * 0.3)
        r = 0.0 * (1-t) + 1.0 * t  # 0 to 1.0
        g = 0.8 * (1-t) + 1.0 * t  # 0.8 to 1.0
        b = 1.0 * (1-t) + 0.0 * t  # 1.0 to 0
        colors_list.append((r, g, b))
    
    # Add yellow to red transition (final 30%)
    for i in range(int(n_colors * 0.3)):
        t = i / (n_colors * 0.3)
        r = 1.0  # Stay at 1.0
        g = 1.0 * (1-t) + 0.0 * t  # 1.0 to 0
        b = 0.0  # Stay at 0
        colors_list.append((r, g, b))
        
    # Create the colormap
    return LinearSegmentedColormap.from_list(name, colors_list, N=n_colors)

def create_depth_colormap(start_depth, end_depth, name='depth_colormap', n_colors=256):
    """
    Create a colormap specifically for depth values.
    
    Args:
        start_depth (float): Minimum depth value.
        end_depth (float): Maximum depth value.
        name (str): Name for the colormap.
        n_colors (int): Number of colors in the colormap.
        
    Returns:
        matplotlib.colors.LinearSegmentedColormap: Depth-specific colormap.
    """
    # Create a colormap that transitions from light to dark with depth
    colors_list = []
    
    for i in range(n_colors):
        t = i / (n_colors - 1)
        h = 0.6 * (1-t)  # Hue: blue (0.6) to purple (0.8)
        s = 0.8  # High saturation throughout
        v = 1.0 - 0.6 * t  # Value decreases with depth (lighter to darker)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors_list.append((r, g, b))
    
    return LinearSegmentedColormap.from_list(name, colors_list, N=n_colors)

def encode_animation_to_html(anim):
    """
    Encode a matplotlib animation as an HTML5 video for display in a notebook.
    
    Args:
        anim (matplotlib.animation.Animation): Animation object to encode.
        
    Returns:
        str: HTML string containing the encoded animation.
    """
    if not hasattr(anim, '_encoded_video'):
        # Save animation
        buffer = python_io.BytesIO()
        anim.save(buffer, writer='pillow', fps=10)
        buffer.seek(0)
        
        # Encode to base64
        video_base64 = base64.b64encode(buffer.read()).decode('ascii')
        anim._encoded_video = f'<img src="data:image/gif;base64,{video_base64}" />'
    
    return HTML(anim._encoded_video)

def display_animation(depths, profiles, title="AFM Depth Profile Animation"):
    """
    Create and display an animation of AFM profiles at different depths.
    
    Args:
        depths (numpy.ndarray): Array of depth values.
        profiles (numpy.ndarray): 2D array of profiles with shape (n_depths, profile_length).
        title (str): Title for the animation.
        
    Returns:
        matplotlib.animation.Animation: Animation object for further manipulation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title, fontsize=16)
    
    # Initial profile plot
    line, = ax.plot(profiles[0], lw=2)
    
    # Set consistent y-limits
    ymin, ymax = profiles.min(), profiles.max()
    padding = (ymax - ymin) * 0.1  # Add 10% padding
    ax.set_ylim(ymin - padding, ymax + padding)
    
    # Add depth indicator
    depth_text = ax.text(0.02, 0.95, f"Depth: {depths[0]:.1f} \u03bcm", transform=ax.transAxes,
                     fontsize=14, bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Spatial coordinate')
    ax.set_ylabel('Modulus (GPa)')
    ax.grid(True)
    
    def update(frame):
        line.set_ydata(profiles[frame])
        depth_text.set_text(f"Depth: {depths[frame]:.1f} \u03bcm")
        return line, depth_text
        
    anim = FuncAnimation(fig, update, frames=len(depths), interval=200, blit=True)
    
    # Display the animation
    plt.close()  # Prevent the empty figure from displaying
    return anim

def enhance_profile_contrast(profiles, percentile_low=2, percentile_high=98):
    """
    Enhance the contrast of AFM profiles for better visualization.
    
    Args:
        profiles (numpy.ndarray): 2D array of profiles.
        percentile_low (float): Lower percentile for contrast stretching.
        percentile_high (float): Higher percentile for contrast stretching.
        
    Returns:
        numpy.ndarray: Enhanced profiles with better contrast.
    """
    # Calculate the percentile values for the entire dataset
    p_low = np.percentile(profiles, percentile_low)
    p_high = np.percentile(profiles, percentile_high)
    
    # Apply contrast stretching
    enhanced = np.clip((profiles - p_low) / (p_high - p_low), 0, 1)
    
    return enhanced

def parse_depth_from_filename(filename):
    """
    Extract depth information from the filename.
    Updated to support optional units like 'um', 'micron', or 'microns'.
    """
    match = re.search(r'(\d+)\s*(?:um|micron(?:s)?)?', filename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Could not parse depth from filename: {filename}")

def prepare_image(image, crop_right=50):
    """
    Crop out the embedded colorbar from the right side of the image.
    The default cropping removes 50 columns from the right. Adjust crop_right as needed.
    """
    if image.shape[1] <= crop_right:
        raise ValueError("crop_right is larger than the image width")
    return image[:, :image.shape[1] - crop_right]

def load_depth_profiles(data_dir, file_pattern="*.jpg", crop_right=50):
    """
    Loads AFM profiles from images in the data directory.
    Crops each image to remove the embedded colorbar, then extracts the central row as the 1D profile
    and associates it with the depth parsed from the filename.
    Returns:
        sorted_depths: list of depths (in microns) sorted in ascending order.
        profiles: 2D numpy array (rows: depth slices, columns: spatial coordinate)
    """
    file_list = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
    if not file_list:
        raise ValueError(f"No files found in {data_dir} with pattern {file_pattern}")

    depth_profiles = []
    depths = []
    for filepath in file_list:
        try:
            depth = parse_depth_from_filename(os.path.basename(filepath))
        except ValueError as e:
            print(e)
            continue

        # Read the image; assuming it is a grayscale image.
        img = io.imread(filepath, as_gray=True)

        # Crop the image to remove the embedded colorbar
        try:
            img_cropped = prepare_image(img, crop_right=crop_right)
        except ValueError as e:
            print(e)
            continue

        # Extract the central row as the 1D profile
        profile = img_cropped[img_cropped.shape[0] // 2, :]
        depths.append(depth)
        depth_profiles.append(profile)

    depths = np.array(depths)
    #Enforce uniform profile lengths
    if len(depth_profiles) > 0:
        min_length = min(len(p) for p in depth_profiles)
        depth_profiles = np.array([p[:min_length] for p in depth_profiles])
    else:
        depth_profiles = np.array([])
    sort_idx = np.argsort(depths)
    sorted_depths = depths[sort_idx]
    return sorted_depths, depth_profiles[sort_idx, :]

def interpolate_depth_profiles(depths, profiles, interp_factor=5, method='linear'):
    """
    Interpolate between AFM depth profiles to create a smoother 3D visualization.
    
    Args:
        depths (numpy.ndarray): Original depths where profiles were measured (in microns).
        profiles (numpy.ndarray): 2D array of profiles with shape (n_depths, profile_length).
        interp_factor (int): Factor by which to increase the number of depth slices.
        method (str): Interpolation method ('linear', 'cubic', etc.).
    
    Returns:
        tuple: (interpolated_depths, interpolated_profiles) where:
            - interpolated_depths is a numpy array of new depths with more points
            - interpolated_profiles is a 2D numpy array of interpolated profiles
    """
    if len(depths) < 2:
        return depths, profiles
    
    # Create a finer grid of depths
    depth_min, depth_max = depths.min(), depths.max()
    num_interp_depths = (len(depths) - 1) * interp_factor + 1
    interpolated_depths = np.linspace(depth_min, depth_max, num_interp_depths)
    
    # Initialize the array for interpolated profiles
    interpolated_profiles = np.zeros((len(interpolated_depths), profiles.shape[1]))
    
    # Interpolate each spatial position across depths
    for col in range(profiles.shape[1]):
        # Extract the profile values at this spatial position for all depths
        depth_profile = profiles[:, col]
        
        # Create an interpolation function for this spatial position
        if method == 'linear' and len(depths) >= 2:
            interpolator = interp1d(depths, depth_profile, kind='linear')
        elif method == 'cubic' and len(depths) >= 4:
            interpolator = interp1d(depths, depth_profile, kind='cubic')
        elif method == 'quadratic' and len(depths) >= 3:
            interpolator = interp1d(depths, depth_profile, kind='quadratic')
        else:
            # Default to linear if not enough points for the requested method
            interpolator = interp1d(depths, depth_profile, kind='linear')
        
        # Apply the interpolation to get values at the new depths
        interpolated_profiles[:, col] = interpolator(interpolated_depths)
    
    return interpolated_depths, interpolated_profiles

def create_comparison_visualization(orig_depths, orig_profiles, interp_depths, interp_profiles):
    """
    Create a visualization that allows toggling between original and interpolated data.
    
    Args:
        orig_depths (numpy.ndarray): Original depths.
        orig_profiles (numpy.ndarray): Original depth profiles.
        interp_depths (numpy.ndarray): Interpolated depths with more points.
        interp_profiles (numpy.ndarray): Interpolated depth profiles.
    """
    # Create a figure with comparison plots
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
    
    # Create subplots
    ax_stack = fig.add_subplot(gs[0, 0])
    ax_profile = fig.add_subplot(gs[0, 1])
    ax_3d = fig.add_subplot(gs[:, 2], projection='3d')
    ax_compare = fig.add_subplot(gs[1, :2])
    
    # We'll toggle between original and interpolated data
    all_depths = [orig_depths, interp_depths]
    all_profiles = [orig_profiles, interp_profiles]
    data_type_names = ['Original', 'Interpolated']
    current_data_idx = 1  # Start with interpolated data
    
    # Initialize visualization with interpolated data (smoother)
    current_depths = interp_depths
    current_profiles = interp_profiles
    
    # Initial stack plot
    stack = ax_stack.imshow(current_profiles, aspect='auto', cmap='viridis',
                         extent=[0, current_profiles.shape[1]-1, current_depths[-1], current_depths[0]])
    ax_stack.set_xlabel("Spatial coordinate")
    ax_stack.set_ylabel("Depth (microns)")
    ax_stack.set_title(f"{data_type_names[current_data_idx]} AFM Tomography Stack")
    cbar_stack = fig.colorbar(stack, ax=ax_stack, label="Modulus (GPa)")
    
    # Initial profile plot
    slice_idx = 0  # Start with first slice
    profile_line, = ax_profile.plot(current_profiles[slice_idx], 'b-', linewidth=2)
    ax_profile.set_xlabel("Spatial coordinate")
    ax_profile.set_ylabel("Modulus (GPa)")
    ax_profile.set_title(f"Profile at Depth: {current_depths[slice_idx]:.1f} µm")
    ax_profile.grid(True)
    
    # Initial 3D plot
    X = np.arange(current_profiles.shape[1])
    Y = current_depths
    X, Y = np.meshgrid(X, Y)
    surf = ax_3d.plot_surface(X, Y, current_profiles, cmap='viridis',
                            edgecolor='none', alpha=0.8)
    ax_3d.set_xlabel('Spatial coordinate')
    ax_3d.set_ylabel('Depth (microns)')
    ax_3d.set_zlabel('Modulus (GPa)')
    ax_3d.set_title(f'{data_type_names[current_data_idx]} 3D AFM Tomography')
    cbar_3d = fig.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=5, label='Modulus (GPa)')
    
    # Initial comparison plot - show both original and interpolated at selected depth
    orig_line, = ax_compare.plot(orig_profiles[0], 'r-', linewidth=2, label='Original')
    interp_line, = ax_compare.plot(interp_profiles[0], 'b-', linewidth=2, label='Interpolated')
    ax_compare.set_xlabel("Spatial coordinate")
    ax_compare.set_ylabel("Modulus (GPa)")
    ax_compare.set_title(f"Comparison at Depth: {orig_depths[0]:.1f} µm")
    ax_compare.grid(True)
    ax_compare.legend()
    
    plt.tight_layout()
    
    # Function to update the visualization when switching between original and interpolated
    def update_data_type(data_type_idx):
        nonlocal current_data_idx, current_depths, current_profiles
        
        current_data_idx = data_type_idx
        current_depths = all_depths[current_data_idx]
        current_profiles = all_profiles[current_data_idx]
        
        # Update the stack plot
        stack.set_data(current_profiles)
        stack.set_extent([0, current_profiles.shape[1]-1, current_depths[-1], current_depths[0]])
        ax_stack.set_title(f"{data_type_names[current_data_idx]} AFM Tomography Stack")
        
        # Update the 3D plot - recreate the surface
        ax_3d.clear()
        X = np.arange(current_profiles.shape[1])
        Y = current_depths
        X, Y = np.meshgrid(X, Y)
        surf = ax_3d.plot_surface(X, Y, current_profiles, cmap='viridis',
                                edgecolor='none', alpha=0.8)
        ax_3d.set_xlabel('Spatial coordinate')
        ax_3d.set_ylabel('Depth (microns)')
        ax_3d.set_zlabel('Modulus (GPa)')
        ax_3d.set_title(f'{data_type_names[current_data_idx]} 3D AFM Tomography')
        
        # Update the slice slider max value
        depth_slider.max = len(current_depths) - 1
        
        # Also update the current slice within the new data type
        update_slice(min(slice_slider.value, len(current_depths)-1))
        
        fig.canvas.draw_idle()
    
    # Function to update when the depth slice changes
    def update_slice(slice_idx):
        if slice_idx >= len(current_depths):
            slice_idx = len(current_depths) - 1
            
        current_depth = current_depths[slice_idx]
        
        # Update the profile plot
        profile_line.set_ydata(current_profiles[slice_idx])
        ax_profile.set_title(f"Profile at Depth: {current_depth:.1f} µm")
        
        # Find closest depths in original and interpolated data
        orig_idx = np.abs(orig_depths - current_depth).argmin()
        interp_idx = np.abs(interp_depths - current_depth).argmin()
        
        # Update the comparison plot
        orig_line.set_ydata(orig_profiles[orig_idx])
        interp_line.set_ydata(interp_profiles[interp_idx])
        ax_compare.set_title(f"Comparison at approx. Depth: {current_depth:.1f} µm")
        
        # Update y-axis limits if needed
        ymin = min(np.min(orig_profiles[orig_idx]), np.min(interp_profiles[interp_idx]))
        ymax = max(np.max(orig_profiles[orig_idx]), np.max(interp_profiles[interp_idx]))
        padding = (ymax - ymin) * 0.1  # Add 10% padding
        ax_profile.set_ylim(ymin - padding, ymax + padding)
        ax_compare.set_ylim(ymin - padding, ymax + padding)
        
        fig.canvas.draw_idle()
    
    # Function to update the 3D view angles
    def update_view(azim, elev):
        ax_3d.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()
    
    # Create sliders for interaction
    data_toggle = widgets.ToggleButtons(
        options=[(data_type_names[0], 0), (data_type_names[1], 1)],
        description='Data:',
        value=1,  # Start with interpolated
        tooltip='Toggle between original and interpolated data'
    )
    
    slice_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(current_depths)-1,
        step=1,
        description='Depth:',
        continuous_update=True
    )
    
    azim_slider = widgets.IntSlider(
        value=30,
        min=0,
        max=360,
        step=5,
        description='Azimuth:',
        continuous_update=True
    )
    
    elev_slider = widgets.IntSlider(
        value=20,
        min=0,
        max=90,
        step=5,
        description='Elevation:',
        continuous_update=True
    )
    
    # Rename for clearer reference in update functions
    depth_slider = slice_slider
    
    # Create the widgets layout
    ui = widgets.VBox([
        widgets.HBox([data_toggle]),
        widgets.HBox([slice_slider]),
        widgets.HBox([azim_slider, elev_slider])
    ])
    
    # Connect the widgets to their update functions
    out1 = widgets.interactive_output(update_data_type, {'data_type_idx': data_toggle})
    out2 = widgets.interactive_output(update_slice, {'slice_idx': slice_slider})
    out3 = widgets.interactive_output(update_view, {'azim': azim_slider, 'elev': elev_slider})
    
    display(ui, out1, out2, out3)
    plt.show()

def dynamic_visualization(depths, profiles):
    # Create a figure with three subplots: stack, profile, and 3D view
    fig = plt.figure(figsize=(18, 10))
    
    # Set up grid for the plots
    gs = fig.add_gridspec(2, 3)
    ax_stack = fig.add_subplot(gs[0, 0])
    ax_profile = fig.add_subplot(gs[0, 1])
    ax_3d = fig.add_subplot(gs[:, 2], projection='3d')
    ax_colormap = fig.add_subplot(gs[1, :2])
    
    # Plot the complete stack
    im = ax_stack.imshow(profiles, aspect='auto', cmap='viridis', 
                         extent=[0, profiles.shape[1]-1, depths[-1], depths[0]])
    ax_stack.set_xlabel("Spatial coordinate")
    ax_stack.set_ylabel("Depth (microns)")
    ax_stack.set_title("AFM Tomography Stack")
    
    # Make a colorbar for the stack plot
    cbar = fig.colorbar(im, ax=ax_stack, label="Modulus (GPa)")
    
    # Initialize horizontal marker and initial 1D profile
    current_line = ax_stack.axhline(depths[0], color='red', linewidth=2, label="Selected Depth")
    ax_stack.legend()
    current_profile, = ax_profile.plot(profiles[0, :], color='blue', linewidth=2)
    ax_profile.set_xlabel("Spatial coordinate")
    ax_profile.set_ylabel("Modulus (GPa)")
    ax_profile.set_title(f"AFM Profile at Depth: {depths[0]:.1f} µm")
    ax_profile.grid(True)
    
    # Create 3D surface plot
    X = np.arange(profiles.shape[1])
    Y = depths
    X, Y = np.meshgrid(X, Y)
    surf = ax_3d.plot_surface(X, Y, profiles, cmap='viridis', 
                             edgecolor='none', alpha=0.8)
    ax_3d.set_xlabel('Spatial coordinate')
    ax_3d.set_ylabel('Depth (microns)')
    ax_3d.set_zlabel('Modulus (GPa)')
    ax_3d.set_title('3D AFM Tomography')
    
    # Add a color bar which maps values to colors
    fig.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=5, label='Modulus (GPa)')
    
    # Create a heatmap view
    heatmap = ax_colormap.imshow(profiles, aspect='auto', cmap='viridis',
                               extent=[0, profiles.shape[1]-1, depths[-1], depths[0]])
    ax_colormap.set_xlabel("Spatial coordinate")
    ax_colormap.set_ylabel("Depth (microns)")
    ax_colormap.set_title("AFM Depth Profile Heatmap")
    fig.colorbar(heatmap, ax=ax_colormap, orientation='horizontal', label="Modulus (GPa)")
    
    # Initial depth marker in the heatmap
    heatmap_line = ax_colormap.axhline(depths[0], color='red', linewidth=2)
    
    # Initial 3D marker plane at the selected depth
    z_min, z_max = np.min(profiles), np.max(profiles)
    x_min, x_max = 0, profiles.shape[1]-1
    selected_depth = depths[0]
    depth_marker = ax_3d.plot([x_min, x_max], [selected_depth, selected_depth], 
                             [z_min, z_min], 'r-', linewidth=2)
    
    plt.tight_layout()
    
    def update_slice(slice_index):
        selected_depth = depths[slice_index]
        
        # Update the marker in the stack view
        current_line.set_ydata([selected_depth, selected_depth])
        ax_stack.set_ylim(depths[-1], depths[0])  # Keep y axis direction
        
        # Update the profile view
        current_profile.set_ydata(profiles[slice_index, :])
        ax_profile.set_title(f"AFM Profile at Depth: {selected_depth:.1f} µm")
        
        # Update the marker in the heatmap
        heatmap_line.set_ydata([selected_depth, selected_depth])
        
        # Update the 3D marker plane
        depth_marker[0].remove()
        depth_marker[0] = ax_3d.plot([x_min, x_max], [selected_depth, selected_depth], 
                                    [z_min, z_min], 'r-', linewidth=2)[0]
        
        # Rotate the 3D view slightly each time for better visualization
        current_azim = ax_3d.azim
        ax_3d.view_init(elev=30, azim=(current_azim + 2) % 360)
        
        fig.canvas.draw_idle()
    
    # Create interactive slider
    slider = widgets.IntSlider(value=0, min=0, max=profiles.shape[0]-1, step=1,
                             description='Depth Slice', continuous_update=True)
    
    # Add rotation controls for 3D plot
    azim_slider = widgets.IntSlider(value=30, min=0, max=360, step=5,
                                   description='Azimuth', continuous_update=True)
    elev_slider = widgets.IntSlider(value=30, min=0, max=90, step=5,
                                   description='Elevation', continuous_update=True)
    
    def update_view(azim, elev):
        ax_3d.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()
    
    # Combined UI
    ui = widgets.VBox([
        widgets.HBox([slider]),
        widgets.HBox([azim_slider, elev_slider])
    ])
    
    # Connect the sliders to their update functions
    out1 = widgets.interactive_output(update_slice, {'slice_index': slider})
    out2 = widgets.interactive_output(update_view, {'azim': azim_slider, 'elev': elev_slider})
    
    display(ui, out1, out2)
    plt.show()
    return fig

def main(crop_right=50, interp_factor=5, interp_method='cubic', use_interpolation=True):
    """
    Main function to load and visualize AFM depth profiles.
    
    Args:
        crop_right (int): Number of pixels to crop from the right of each image.
        interp_factor (int): Factor by which to increase the number of depth slices.
        interp_method (str): Interpolation method ('linear', 'cubic', 'quadratic').
        use_interpolation (bool): Whether to use interpolation for smoother 3D visualization.
    """
    data_dir = os.path.join("data")
    file_pattern = "*.jpg"  # Updated to JPG files
    file_list = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
    
    if not file_list:
        print(f"No files found in {data_dir} with pattern {file_pattern}")
        return

    # Load and display the first four images
    num_images_to_display = min(4, len(file_list))
    fig_images, axes_images = plt.subplots(1, num_images_to_display, figsize=(15, 5))
    
    # Handle the case where there's only one image
    if num_images_to_display == 1:
        axes_images = [axes_images]

    for i in range(num_images_to_display):
        filepath = file_list[i]
        img = io.imread(filepath, as_gray=True)
        axes_images[i].imshow(img, cmap='gray')
        axes_images[i].set_title(os.path.basename(filepath))
        axes_images[i].axhline(img.shape[0] // 2, color='red', linewidth=1) # Show central row
        axes_images[i].axis('off')

    plt.tight_layout()
    plt.show()

    try:
        depths, profiles = load_depth_profiles(data_dir, file_pattern="*.jpg", crop_right=crop_right)
    except ValueError as e:
        print(e)
        return
    
    # If there aren't enough points for cubic interpolation, default to linear
    if interp_method == 'cubic' and len(depths) < 4:
        print("Not enough depth points for cubic interpolation. Using linear instead.")
        interp_method = 'linear'
    elif interp_method == 'quadratic' and len(depths) < 3:
        print("Not enough depth points for quadratic interpolation. Using linear instead.")
        interp_method = 'linear'
    
    print("Loaded depths (microns):", depths)
    
    if use_interpolation:
        print(f"Applying {interp_method} interpolation with factor {interp_factor}...")
        interp_depths, interp_profiles = interpolate_depth_profiles(
            depths, profiles, interp_factor=interp_factor, method=interp_method)
        print(f"Interpolated: {len(depths)} slices → {len(interp_depths)} slices")
        # Visualize using the interpolated data
        dynamic_visualization(interp_depths, interp_profiles)
        
        # Also create a comparison visualization that toggles between original and interpolated
        create_comparison_visualization(depths, profiles, interp_depths, interp_profiles)
    else:
        # Visualize using the original data only
        dynamic_visualization(depths, profiles)

if __name__ == "__main__":
    main()
