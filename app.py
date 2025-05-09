import os
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import plotly.express as px
import logging

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import our custom modules
from data_loader import load_depth_profiles
from data_processor import enhance_profile_contrast, interpolate_depth_profiles, create_depth_matrix
from colormaps import generate_plotly_colorscale
from visualizations import create_profile_plot, create_heatmap_plot, create_3d_surface_plot
from ml.integration import MLIntegration

# Initialize the Dash app with Bootstrap styling
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    suppress_callback_exceptions=True
)  # Single theme, stable app


# Initialize ML integration
ml_integration = MLIntegration(app)

app.title = "DepthScan - AFM Tomography Analysis Tool"
server = app.server  # Expose Flask server for production deployment

# Define the app layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Img(src="/assets/DScan-Logo-2.png", style={"height": "80px", "display": "block", "margin": "0 auto", "paddingTop": "16px"}),
                        html.H1("DepthScan", className="text-center my-4"),
                        html.H4("Advanced AFM Tomography Analysis Tool", className="text-center text-muted mb-4"),
                    ],
                    width=12,
                ),
            ]
        ),
        
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Data Settings"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Data Directory"),
                                                        dcc.Dropdown(
                                                            id="data-directory",
                                                            options=[
                                                                {"label": "Data Set 1", "value": "data"},
                                                                {"label": "Data Set 2", "value": "data-2"},
                                                            ],
                                                            value="data",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("File Pattern"),
                                                        dcc.Input(
                                                            id="file-pattern",
                                                            type="text",
                                                            value="*.jpg",
                                                            className="form-control",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Crop Right (pixels)"),
                                                        dcc.Slider(
                                                            id="crop-right",
                                                            min=0,
                                                            max=100,
                                                            step=5,
                                                            value=50,
                                                            marks={i: str(i) for i in range(0, 101, 20)},
                                                        ),
                                                    ],
                                                    width=12,
                                                ),
                                            ]
                                        ),
                                        html.Br(),
                                        dbc.Button(
                                            "Load Data",
                                            id="load-data-button",
                                            color="primary",
                                            className="w-100",
                                        ),
                                        html.Div(id="data-loading-output"),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                        
                        dbc.Card(
                            [
                                dbc.CardHeader("Processing Settings"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Use Interpolation"),
                                                        dbc.Switch(
                                                            id="use-interpolation",
                                                            value=True,
                                                        ),
                                                    ],
                                                    width=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Interpolation Factor"),
                                                        dcc.Slider(
                                                            id="interp-factor",
                                                            min=1,
                                                            max=10,
                                                            step=1,
                                                            value=5,
                                                            marks={i: str(i) for i in range(1, 11)},
                                                        ),
                                                    ],
                                                    width=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Interpolation Method"),
                                                        dcc.Dropdown(
                                                            id="interp-method",
                                                            options=[
                                                                {"label": "Linear", "value": "linear"},
                                                                {"label": "Cubic", "value": "cubic"},
                                                                {"label": "Quadratic", "value": "quadratic"},
                                                            ],
                                                            value="cubic",
                                                        ),
                                                    ],
                                                    width=4,
                                                ),
                                            ]
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Contrast Enhancement"),
                                                        dbc.Switch(
                                                            id="use-contrast-enhancement",
                                                            value=True,
                                                        ),
                                                    ],
                                                    width=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Low Percentile"),
                                                        dcc.Slider(
                                                            id="percentile-low",
                                                            min=0,
                                                            max=20,
                                                            step=1,
                                                            value=2,
                                                            marks={i: str(i) for i in range(0, 21, 5)},
                                                        ),
                                                    ],
                                                    width=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("High Percentile"),
                                                        dcc.Slider(
                                                            id="percentile-high",
                                                            min=80,
                                                            max=100,
                                                            step=1,
                                                            value=98,
                                                            marks={i: str(i) for i in range(80, 101, 5)},
                                                        ),
                                                    ],
                                                    width=4,
                                                ),
                                            ]
                                        ),
                                        html.Br(),
                                        dbc.Button(
                                            "Process Data",
                                            id="process-data-button",
                                            color="success",
                                            className="w-100",
                                            disabled=True,
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                    ],
                    width=12, md=4,
                ),
                
                dbc.Col(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    [
                                        dcc.Graph(id="profile-plot", style={"height": "60vh"}),
                                        html.Div(
                                            [
                                                html.Label("Select Depth"),
                                                dcc.Slider(
                                                    id="depth-slider",
                                                    min=0,
                                                    max=10,
                                                    step=0.1,
                                                    value=0,
                                                    marks={i: f"{i}μm" for i in range(0, 11, 2)},
                                                ),
                                            ],
                                            className="mt-4",
                                        ),
                                    ],
                                    label="Profile View",
                                ),
                                dbc.Tab(
                                    dcc.Graph(id="heatmap-plot", style={"height": "70vh"}),
                                    label="Heatmap View",
                                ),
                                dbc.Tab(
                                    dcc.Graph(id="surface-plot", style={"height": "70vh"}),
                                    label="3D Surface",
                                ),
                            ]
                        ),
                    ],
                    width=12, md=8,
                ),
            ]
        ),
        
        # Add ML components
        *ml_integration.get_ml_components(),
        
        # Store components for intermediate data
        dcc.Store(id="original-data-store"),
        dcc.Store(id="processed-data-store"),
        
        # Footer
        dbc.Row(
            dbc.Col(
                html.P(
                    "DepthScan - Advanced AFM Tomography Analysis Tool | 2025",
                    className="text-center text-muted mt-4",
                ),
                width=12,
            )
        ),
    ],
    fluid=True,
    className="px-4 py-3",
)

# Callback to load data
@app.callback(
    [
        Output("original-data-store", "data"),
        Output("data-loading-output", "children"),
        Output("process-data-button", "disabled"),
        Output("depth-slider", "min"),
        Output("depth-slider", "max"),
        Output("depth-slider", "marks"),
        Output("depth-slider", "value"),
    ],
    [
        Input("load-data-button", "n_clicks"),
    ],
    [
        State("data-directory", "value"),
        State("file-pattern", "value"),
        State("crop-right", "value"),
    ],
    prevent_initial_call=True,
)
def load_data(n_clicks, data_dir, file_pattern, crop_right):
    if n_clicks is None:
        return dash.no_update
    
    try:
        # Load the data
        depths, profiles = load_depth_profiles(data_dir, file_pattern, crop_right)
        
        # Create marks for the depth slider
        min_depth = depths.min()
        max_depth = depths.max()
        step = max(1, int((max_depth - min_depth) / 5))  # Create about 5 marks
        marks = {i: f"{i}μm" for i in range(int(min_depth), int(max_depth) + 1, step)}
        
        # Store the data
        data = {
            "depths": depths.tolist(),
            "profiles": profiles.tolist(),
        }
        
        return (
            data,
            html.Div(f"Loaded {len(depths)} depth profiles", className="text-success mt-2"),
            False,  # Enable the process button
            min_depth,
            max_depth,
            marks,
            min_depth,
        )
    except Exception as e:
        return (
            dash.no_update,
            html.Div(f"Error: {str(e)}", className="text-danger mt-2"),
            True,  # Keep process button disabled
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

# Callback to process data
@app.callback(
    Output("processed-data-store", "data"),
    [
        Input("process-data-button", "n_clicks"),
    ],
    [
        State("original-data-store", "data"),
        State("use-interpolation", "value"),
        State("interp-factor", "value"),
        State("interp-method", "value"),
        State("use-contrast-enhancement", "value"),
        State("percentile-low", "value"),
        State("percentile-high", "value"),
    ],
    prevent_initial_call=True,
)
def process_data(n_clicks, data, use_interpolation, interp_factor, interp_method,
                use_contrast_enhancement, percentile_low, percentile_high):
    if n_clicks is None or data is None:
        return dash.no_update
    
    # Convert data back to numpy arrays
    depths = np.array(data["depths"])
    profiles = np.array(data["profiles"])
    
    # Apply contrast enhancement if requested
    if use_contrast_enhancement:
        profiles = enhance_profile_contrast(profiles, percentile_low, percentile_high)
    
    # Apply interpolation if requested
    if use_interpolation:
        interp_depths, interp_profiles = interpolate_depth_profiles(
            depths, profiles, interp_factor, interp_method
        )
    else:
        interp_depths, interp_profiles = depths, profiles
    
    # Store the processed data
    processed_data = {
        "depths": interp_depths.tolist(),
        "profiles": interp_profiles.tolist(),
        "original_depths": depths.tolist(),
        "original_profiles": profiles.tolist(),
    }
    
    return processed_data

# Callback to update the profile plot
@app.callback(
    Output("profile-plot", "figure"),
    [
        Input("processed-data-store", "data"),
        Input("depth-slider", "value"),
    ],
    prevent_initial_call=True,
)
def update_profile_plot(data, selected_depth):
    if data is None:
        return go.Figure().update_layout(title="No data available")
    
    # Convert data back to numpy arrays
    depths = np.array(data["depths"])
    profiles = np.array(data["profiles"])
    
    # Find the closest depth to the selected depth
    closest_idx = np.abs(depths - selected_depth).argmin()
    
    # Create the figure
    fig = go.Figure()
    
    # Add the profile trace
    fig.add_trace(go.Scatter(
        x=np.arange(len(profiles[closest_idx])),
        y=profiles[closest_idx],
        mode="lines",
        name=f"Depth: {depths[closest_idx]:.1f}μm",
        line=dict(color="#1f77b4", width=2),
    ))
    
    # Update layout
    fig.update_layout(
        title=f"AFM Profile at Depth {depths[closest_idx]:.1f}μm",
        xaxis_title="Spatial Coordinate",
        yaxis_title="Normalized Signal",
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
    )
    
    return fig

# Callback to update the heatmap plot
@app.callback(
    Output("heatmap-plot", "figure"),
    [
        Input("processed-data-store", "data"),
    ],
    prevent_initial_call=True,
)
def update_heatmap_plot(data):
    if data is None:
        return go.Figure().update_layout(title="No data available")
    
    # Convert data back to numpy arrays
    depths = np.array(data["depths"])
    profiles = np.array(data["profiles"])
    
    # Create the figure
    fig = go.Figure()
    
    # Add the heatmap trace
    fig.add_trace(go.Heatmap(
        z=profiles,
        x=np.arange(profiles.shape[1]),
        y=depths,
        colorscale=generate_plotly_colorscale("afm"),
        colorbar=dict(title="Signal"),
    ))
    
    # Update layout
    fig.update_layout(
        title="AFM Depth Profiles Heatmap",
        xaxis_title="Spatial Coordinate",
        yaxis_title="Depth (μm)",
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
    )
    
    return fig

# Callback to update the 3D surface plot
@app.callback(
    Output("surface-plot", "figure"),
    [
        Input("processed-data-store", "data"),
    ],
    prevent_initial_call=True,
)
def update_surface_plot(data):
    if data is None:
        return go.Figure().update_layout(title="No data available")
    
    # Convert data back to numpy arrays
    depths = np.array(data["depths"])
    profiles = np.array(data["profiles"])
    
    # Create the figure
    fig = go.Figure()
    
    # Add the surface trace
    fig.add_trace(go.Surface(
        z=profiles,
        x=np.arange(profiles.shape[1]),
        y=depths,
        colorscale=generate_plotly_colorscale("afm"),
        colorbar=dict(title="Signal"),
    ))
    
    # Update layout
    fig.update_layout(
        title="AFM Depth Profiles 3D Surface",
        scene=dict(
            xaxis_title="Spatial Coordinate",
            yaxis_title="Depth (μm)",
            zaxis_title="Signal",
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2),
            ),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_white",
    )
    
    return fig

# Run the app

if __name__ == "__main__":
    app.run(debug=True, port=8053)
def get_config(self):
    """
    Return the current feature extraction configuration.
    You can expand this as needed to include more settings.
    """
    return {
        "available_methods": getattr(self, "available_methods", []),
        "n_components": getattr(self, "n_components", None)
    }