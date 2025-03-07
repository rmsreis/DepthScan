import numpy as np
import plotly.graph_objs as go
from colormaps import generate_plotly_colorscale

def create_profile_plot(profile, depth, x_coords=None):
    """
    Create a Plotly figure for a single depth profile.
    
    Args:
        profile (numpy.ndarray): 1D array representing the profile at a specific depth
        depth (float): The depth value in microns
        x_coords (numpy.ndarray, optional): Spatial coordinates for the x-axis
        
    Returns:
        plotly.graph_objs.Figure: Plotly figure object for the profile plot
    """
    if x_coords is None:
        x_coords = np.arange(len(profile))
    
    fig = go.Figure()
    
    # Add the profile trace
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=profile,
        mode="lines",
        name=f"Depth: {depth:.1f}u03bcm",
        line=dict(color="#1f77b4", width=2),
    ))
    
    # Update layout
    fig.update_layout(
        title=f"AFM Profile at Depth {depth:.1f}u03bcm",
        xaxis_title="Spatial Coordinate",
        yaxis_title="Normalized Signal",
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
    )
    
    return fig

def create_heatmap_plot(depths, profiles, x_coords=None):
    """
    Create a Plotly heatmap figure for AFM depth profiles.
    
    Args:
        depths (numpy.ndarray): 1D array of depth values
        profiles (numpy.ndarray): 2D array of profiles with shape (n_depths, profile_length)
        x_coords (numpy.ndarray, optional): Spatial coordinates for the x-axis
        
    Returns:
        plotly.graph_objs.Figure: Plotly figure object for the heatmap plot
    """
    if x_coords is None:
        x_coords = np.arange(profiles.shape[1])
    
    fig = go.Figure()
    
    # Add the heatmap trace
    fig.add_trace(go.Heatmap(
        z=profiles,
        x=x_coords,
        y=depths,
        colorscale=generate_plotly_colorscale("afm"),
        colorbar=dict(title="Signal"),
    ))
    
    # Update layout
    fig.update_layout(
        title="AFM Depth Profiles Heatmap",
        xaxis_title="Spatial Coordinate",
        yaxis_title="Depth (u03bcm)",
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
    )
    
    return fig

def create_3d_surface_plot(depths, profiles, x_coords=None):
    """
    Create a Plotly 3D surface plot for AFM depth profiles.
    
    Args:
        depths (numpy.ndarray): 1D array of depth values
        profiles (numpy.ndarray): 2D array of profiles with shape (n_depths, profile_length)
        x_coords (numpy.ndarray, optional): Spatial coordinates for the x-axis
        
    Returns:
        plotly.graph_objs.Figure: Plotly figure object for the 3D surface plot
    """
    if x_coords is None:
        x_coords = np.arange(profiles.shape[1])
    
    fig = go.Figure()
    
    # Add the surface trace
    fig.add_trace(go.Surface(
        z=profiles,
        x=x_coords,
        y=depths,
        colorscale=generate_plotly_colorscale("afm"),
        colorbar=dict(title="Signal"),
    ))
    
    # Update layout
    fig.update_layout(
        title="AFM Depth Profiles 3D Surface",
        scene=dict(
            xaxis_title="Spatial Coordinate",
            yaxis_title="Depth (u03bcm)",
            zaxis_title="Signal",
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2),
            ),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_white",
    )
    
    return fig
