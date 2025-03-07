import colorsys
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def create_custom_colormap(name='afm_colormap', n_colors=256):
    """
    Create a custom colormap optimized for AFM data visualization.
    
    Args:
        name (str): Name for the custom colormap
        n_colors (int): Number of colors in the colormap
        
    Returns:
        matplotlib.colors.LinearSegmentedColormap: Custom colormap object
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
        start_depth (float): Minimum depth value
        end_depth (float): Maximum depth value
        name (str): Name for the colormap
        n_colors (int): Number of colors in the colormap
        
    Returns:
        matplotlib.colors.LinearSegmentedColormap: Depth-specific colormap
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

def generate_plotly_colorscale(colormap_name='afm'):
    """
    Generate a Plotly colorscale from a custom colormap.
    
    Args:
        colormap_name (str): Name of the colormap to use ('afm' or 'depth')
        
    Returns:
        list: Plotly colorscale as a list of [position, color] pairs
    """
    if colormap_name == 'afm':
        cmap = create_custom_colormap()
    elif colormap_name == 'depth':
        cmap = create_depth_colormap(0, 10)
    else:
        raise ValueError(f"Unknown colormap name: {colormap_name}")
    
    # Create a colorscale for Plotly
    positions = np.linspace(0, 1, 256)
    colors = [cmap(pos) for pos in positions]
    rgb_strings = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b, _ in colors]
    
    return [[pos, color] for pos, color in zip(positions, rgb_strings)]
