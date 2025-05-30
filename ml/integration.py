import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import logging

logger = logging.getLogger(__name__)

# Import our ML modules
from .feature_extraction import FeatureExtractor, ClusteringAnalysis
from .anomaly_detection import AnomalyDetector, AnomalyAnalyzer
from .model_trainer import ModelTrainer
from .visualizations import MLVisualizer
from .inference import ModelInference

class MLIntegration:
    """
    Integrate machine learning capabilities into the DepthScan application.
    
    This class provides methods to:
    - Add ML controls to the UI
    - Handle ML-related callbacks
    - Manage ML model states
    - Display ML results
    """
    
    def __init__(self, app):
        """
        Initialize the ML integration.
        
        Args:
            app (dash.Dash): The Dash application instance
        """
        self.app = app
        logger.info("Initializing ML Integration")
        
        # Initialize unsupervised ML components
        self.feature_extractor = FeatureExtractor()
        self.clustering_analyzer = ClusteringAnalysis()
        self.anomaly_detector = AnomalyDetector()
        self.anomaly_analyzer = AnomalyAnalyzer(self.anomaly_detector)
        
        # Initialize model trainer and visualizer
        self.model_trainer = ModelTrainer()
        self.visualizer = MLVisualizer()
        
        # Initialize state variables
        self.extracted_features = None
        self.feature_extraction_method = None
        self.clustering_labels = None
        self.anomaly_labels = None
        
        # Register ML callbacks
        self._register_callbacks()
    
    def get_ml_components(self):
        """Return ML components to be added to the app layout."""
        # Create ML tab
        ml_tab = dbc.Tab(
            label="Machine Learning",
            tab_id="ml-tab",
            children=[
                dbc.Row([
                    dbc.Col([
                        # Feature Extraction Card
                        dbc.Card([
                            dbc.CardHeader("Feature Extraction"),
                            dbc.CardBody([
                                html.Label("Feature Extraction Method"),
                                dcc.Dropdown(
                                    id='feature-extraction-method',
                                    options=[
                                        {'label': 'PCA', 'value': 'pca'},
                                        {'label': 'Kernel PCA', 'value': 'kernel_pca'},
                                        {'label': 'NMF', 'value': 'nmf'},
                                        {'label': 'ICA', 'value': 'ica'},
                                        {'label': 't-SNE', 'value': 'tsne'},
                                        {'label': 'Isomap', 'value': 'isomap'},
                                        {'label': 'LLE', 'value': 'lle'}
                                    ],
                                    value='pca'
                                ),
                                html.Br(),
                                html.Label("Number of Components"),
                                dcc.Slider(
                                    id='n-components-slider',
                                    min=2,
                                    max=10,
                                    step=1,
                                    value=2,
                                    marks={i: str(i) for i in range(2, 11, 2)}
                                ),
                                html.Br(),
                                dbc.Button(
                                    "Extract Features",
                                    id="extract-features-button",
                                    color="primary",
                                    className="w-100"
                                ),
                                html.Div(id="feature-extraction-output", className="mt-2")
                            ])
                        ], className="mb-4"),
                        
                        # Clustering Card
                        dbc.Card([
                            dbc.CardHeader("Clustering"),
                            dbc.CardBody([
                                html.Label("Clustering Method"),
                                dcc.Dropdown(
                                    id='clustering-method',
                                    options=[
                                        {'label': 'K-Means', 'value': 'kmeans'},
                                        {'label': 'DBSCAN', 'value': 'dbscan'},
                                        {'label': 'Spectral Clustering', 'value': 'spectral'}
                                    ],
                                    value='kmeans'
                                ),
                                html.Br(),
                                html.Div(
                                    id='clustering-params',
                                    children=[
                                        html.Label("Number of Clusters (K-Means)"),
                                        dcc.Slider(
                                            id='n-clusters-slider',
                                            min=2,
                                            max=10,
                                            step=1,
                                            value=3,
                                            marks={i: str(i) for i in range(2, 11, 2)}
                                        )
                                    ]
                                ),
                                html.Br(),
                                dbc.Button(
                                    "Run Clustering",
                                    id="run-clustering-button",
                                    color="primary",
                                    className="w-100"
                                ),
                                html.Div(id="clustering-output", className="mt-2")
                            ])
                        ], className="mb-4"),
                        
                        # Anomaly Detection Card
                        dbc.Card([
                            dbc.CardHeader("Anomaly Detection"),
                            dbc.CardBody([
                                html.Label("Anomaly Detection Method"),
                                dcc.Dropdown(
                                    id='anomaly-detection-method',
                                    options=[
                                        {'label': 'Isolation Forest', 'value': 'isolation_forest'},
                                        {'label': 'Local Outlier Factor', 'value': 'local_outlier_factor'},
                                        {'label': 'One-Class SVM', 'value': 'one_class_svm'},
                                        {'label': 'Robust Covariance', 'value': 'robust_covariance'}
                                    ],
                                    value='isolation_forest'
                                ),
                                html.Br(),
                                html.Label("Contamination (expected % of outliers)"),
                                dcc.Slider(
                                    id='contamination-slider',
                                    min=0.01,
                                    max=0.3,
                                    step=0.01,
                                    value=0.1,
                                    marks={i/100: f"{i}%" for i in range(5, 31, 5)}
                                ),
                                html.Br(),
                                dbc.Button(
                                    "Detect Anomalies",
                                    id="detect-anomalies-button",
                                    color="primary",
                                    className="w-100"
                                ),
                                html.Div(id="anomaly-detection-output", className="mt-2")
                            ])
                        ], className="mb-4")
                    ], width=12, md=4),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H5("ML Results", className="mb-0")
                            ]),
                            dbc.CardBody([
                                dbc.Tabs([
                                    dbc.Tab(
                                        label="Feature Space",
                                        tab_id="feature-space-tab",
                                        children=[
                                            dbc.Row([
                                                dbc.Col([
                                                    dcc.Dropdown(
                                                        id="feature-space-method",
                                                        options=[
                                                            {"label": "PCA", "value": "pca"},
                                                            {"label": "t-SNE", "value": "tsne"}
                                                        ],
                                                        value="pca",
                                                        clearable=False,
                                                        style={"width": "200px"}
                                                    ),
                                                ], width=3),
                                                dbc.Col([
                                                    dbc.Switch(
                                                        id="feature-space-animate",
                                                        label="Animate by Depth",
                                                        value=False,
                                                        style={"marginTop": "8px"}
                                                    ),
                                                ], width=3)
                                            ], className="mb-2"),
                                            dcc.Graph(id='feature-space-graph', style={'height': '600px'})
                                        ]
                                    ),
                                    dbc.Tab(
                                        label="PCA Components",
                                        tab_id="pca-components-tab",
                                        children=[
                                            dbc.Row([
                                                dbc.Col([
                                                    html.Label("Number of Components"),
                                                    dcc.Slider(
                                                        id="pca-components-slider",
                                                        min=2,
                                                        max=8,
                                                        step=2,
                                                        value=4,
                                                        marks={i: str(i) for i in range(2, 9, 2)}
                                                    ),
                                                ], width=6)
                                            ], className="mb-2"),
                                            dcc.Graph(id='pca-components-graph', style={'height': '600px'})
                                        ]
                                    ),
                                    dbc.Tab(
                                        label="Feature Visualization",
                                        tab_id="feature-viz-tab",
                                        children=[
                                            dcc.Graph(id='feature-viz-graph', style={'height': '600px'})
                                        ]
                                    ),
                                    dbc.Tab(
                                        label="Clustering Results",
                                        tab_id="clustering-viz-tab",
                                        children=[
                                            dcc.Graph(id='clustering-viz-graph', style={'height': '600px'})
                                        ]
                                    ),
                                    dbc.Tab(
                                        label="Anomaly Detection",
                                        tab_id="anomaly-viz-tab",
                                        children=[
                                            dcc.Graph(id='anomaly-viz-graph', style={'height': '600px'})
                                        ]
                                    )
                                ], id="ml-results-tabs", active_tab="feature-space-tab")
                            ])
                        ])
                    ], width=8)
                ])
            ])
        
        # Create stores for ML data
        extracted_features_store = dcc.Store(id='extracted-features-store')
        clustering_results_store = dcc.Store(id='clustering-results-store')
        anomaly_results_store = dcc.Store(id='anomaly-results-store')
        
        return [ml_tab, extracted_features_store, clustering_results_store, anomaly_results_store]
    
    def _register_callbacks(self):
        """Register ML-related callbacks."""
        @self.app.callback(
            Output('extracted-features-store', 'data'),
            Output('feature-extraction-output', 'children'),
            Input('extract-features-button', 'n_clicks'),
            State('processed-data-store', 'data'),
            State('feature-types', 'value'),
            prevent_initial_call=True
        )
        def extract_features(n_clicks, processed_data, feature_types):
            if n_clicks is None or processed_data is None:
                raise PreventUpdate
            
            # Convert processed data back to numpy array
            profiles = np.array(processed_data['profiles'])
            
            # Extract features based on selected types
            features = []
            if 'statistical' in feature_types:
                features.append(self.feature_extractor.extract_statistical_features(profiles))
            if 'spectral' in feature_types:
                features.append(self.feature_extractor.extract_spectral_features(profiles))
            if 'morphological' in feature_types:
                features.append(self.feature_extractor.extract_morphological_features(profiles))
            
            # Combine features
            combined_features = np.concatenate(features, axis=-1)
            
            # Convert to list for JSON serialization
            features_list = combined_features.tolist()
            
            return features_list, "Features extracted successfully!"
        
        # --- Feature Space Visualization Callback ---
        @self.app.callback(
            Output('feature-space-graph', 'figure'),
            Input('extracted-features-store', 'data'),
            Input('clustering-results-store', 'data'),
            Input('feature-space-method', 'value'),
            Input('feature-space-animate', 'value'),
            State('processed-data-store', 'data'),
            prevent_initial_call=True
        )
        def update_feature_space_graph(features_data, clustering_data, method, animate, processed_data):
            if features_data is None:
                raise PreventUpdate
            
            features = np.array(features_data)
            
            # Get labels if available
            labels = None
            if clustering_data is not None and 'labels' in clustering_data:
                labels = np.array(clustering_data['labels'])
            
            # Get depths for animation if requested
            frame_column = None
            if animate and processed_data is not None and 'depths' in processed_data:
                frame_column = np.array(processed_data['depths'])
            
            # Use MLVisualizer
            from .visualizations import MLVisualizer
            visualizer = MLVisualizer()
            
            # Create the visualization
            fig = visualizer.plot_dimensionality_reduction(
                features,
                labels=labels,
                method=method,
                n_components=2,
                title=f"{method.upper()} Feature Space",
                animate=animate,
                frame_column=frame_column
            )
            
            return fig
        
        # --- PCA Components Visualization Callback ---
        @self.app.callback(
            Output('pca-components-graph', 'figure'),
            Input('extracted-features-store', 'data'),
            Input('pca-components-slider', 'value'),
            State('processed-data-store', 'data'),
            prevent_initial_call=True
        )
        def update_pca_components_graph(features_data, n_components, processed_data):
            logger.debug(f"PCA Components: Callback triggered. n_components: {n_components}")
            if features_data is None:
                logger.warning("PCA Components: features_data is None. Preventing update.")
                raise PreventUpdate
            
            features = np.array(features_data)
            logger.debug(f"PCA Components: features converted to array with shape {features.shape}")
            
            profile_shape = None
            if processed_data is not None and 'profiles' in processed_data:
                profiles = np.array(processed_data['profiles'])
                logger.debug(f"PCA Components: 'profiles' found in processed_data with shape {profiles.shape} and ndim {profiles.ndim}")
                if profiles.ndim == 2: # (N_profiles, Length_of_profile)
                    profile_shape = (1, profiles.shape[1]) 
                    logger.debug(f"PCA Components: Determined profile_shape for 2D profiles: {profile_shape}")
                elif profiles.ndim > 2: # (N_images, H, W, [C]) or (H, W, [C])
                    profile_shape = profiles.shape[-2:] # Assumes H, W are the last two
                    logger.debug(f"PCA Components: Determined profile_shape for >2D profiles: {profile_shape}")
                else:
                    logger.warning(f"PCA Components: 'profiles' ndim is {profiles.ndim}. Cannot determine profile_shape.")
            elif processed_data is None:
                logger.warning("PCA Components: processed_data is None. Cannot determine profile_shape.")
            else:
                logger.warning("PCA Components: 'profiles' key not found in processed_data. Cannot determine profile_shape.")
            
            try:
                logger.info(f"PCA Components: Calling plot_pca_components with n_components={n_components}, image_shape={profile_shape}")
                fig = self.visualizer.plot_pca_components(
                    features,
                    n_components=n_components,
                    profile_shape=profile_shape,
                    title=f"PCA Components in Image Space ({n_components} components)"
                )
                logger.info("PCA Components: plot_pca_components call successful.")
                return fig
            except Exception as e:
                logger.error(f"Error in PCA visualization: {e}", exc_info=True)
                fig = go.Figure()
                fig.update_layout(
                    title="PCA Components Visualization Error",
                    annotations=[dict(text=f"Error generating PCA components: {str(e)}", showarrow=False)]
                )
                return fig

        @self.app.callback(
            Output('trained-model-store', 'data'),
            Output('model-training-output', 'children'),
            Input('train-model-button', 'n_clicks'),
            State('extracted-features-store', 'data'),
            State('label-data-store', 'data'),
            State('model-type', 'value'),
            prevent_initial_call=True
        )
        def train_model(n_clicks, features_data, label_data, model_type):
            if n_clicks is None or features_data is None:
                raise PreventUpdate
            
            # Convert features back to numpy array
            features = np.array(features_data)
            
            # Store feature extraction configuration
            feature_config = self.feature_extractor.get_config()
            
            # Check for required labels
            if model_type == 'classification' and label_data is None:
                return None, "Error: Label data required for classification"
            
            # Train model based on selected type
            if model_type == 'classification':
                labels = np.array(label_data)
                model_data = self.model_trainer.train_classifier(features, labels)
            elif model_type == 'clustering':
                model_data = self.model_trainer.train_clustering(features)
            elif model_type == 'anomaly_detection':
                model_data = self.model_trainer.train_anomaly_detector(features)
            else:
                return None, f"Error: Unknown model type {model_type}"
            
            # Add feature configuration to model data
            model_data['feature_config'] = feature_config
            
            # Return only serializable data
            serializable_data = {
                'model_path': model_data['model_path'],
                'metrics': model_data['metrics'],
                'labels': model_data['labels'].tolist() if 'labels' in model_data else None,
                'feature_config': feature_config
            }
            
            # Save model
            self.current_model = ModelInference(model_data['model_path'])
            
            return serializable_data, f"Model trained successfully! Metrics: {model_data['metrics']}"
        
        @self.app.callback(
            Output('ml-results-graph', 'figure'),
            Input('trained-model-store', 'data'),
            State('processed-data-store', 'data'),
            prevent_initial_call=True
        )
        def update_ml_results(model_data, processed_data):
            if model_data is None or processed_data is None:
                raise PreventUpdate
            
            # Get features and profiles
            profiles = np.array(processed_data['profiles'])
            features = self.feature_extractor.extract_all_features(profiles)
            
            # Get model type
            model_type = self.current_model.get_model_type()
            
            try:
                # Generate appropriate visualization
                if model_type == 'classifier':
                    predictions = self.current_model.classify(features)
                    fig = self.visualizer.plot_confusion_matrix(
                        np.zeros_like(predictions),  # Placeholder for true labels
                        predictions
                    )
                elif model_type == 'clustering':
                    cluster_assignments = self.current_model.cluster(features)
                    # Use dimensionality reduction to visualize clusters instead of heatmap
                    fig = self.visualizer.plot_dimensionality_reduction(
                        features,
                        labels=cluster_assignments,
                        method='pca',
                        n_components=2,
                        title='Cluster Assignments in Feature Space'
                    )
                elif model_type == 'anomaly_detector':
                    anomalies, scores = self.current_model.detect_anomalies(features)
                    # Use dimensionality reduction to visualize anomalies
                    # Convert anomalies to a categorical variable for coloring
                    anomaly_labels = np.where(anomalies == -1, 'Anomaly', 'Normal')
                    fig = self.visualizer.plot_dimensionality_reduction(
                        features,
                        labels=anomaly_labels,
                        method='pca',
                        n_components=2,
                        title='Anomaly Detection in Feature Space'
                    )
                else:
                    fig = go.Figure()
                    fig.update_layout(
                        title="Unsupported Model Type",
                        annotations=[
                            dict(
                                text="Model type not supported for visualization",
                                xref="paper", yref="paper",
                                x=0.5, y=0.5, showarrow=False
                            )
                        ]
                    )
            except Exception as e:
                fig = go.Figure()
                fig.update_layout(
                    title="Visualization Error",
                    annotations=[
                        dict(
                            text=f"Error generating visualization: {str(e)}",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False
                        )
                    ]
                )
            
            return fig
    
    def get_ml_components(self):
        """
        Get all ML-related components for the app layout.
        
        Returns:
            list: List of ML components
        """
        return [
            dcc.Store(id='extracted-features-store'),
            dcc.Store(id='trained-model-store'),
            dcc.Store(id='label-data-store'),
            dbc.Tab(
                label="Machine Learning",
                children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Feature Extraction"),
                                dbc.CardBody([
                                    html.Label("Feature Types"),
                                    dcc.Checklist(
                                        id='feature-types',
                                        options=[
                                            {'label': 'Statistical', 'value': 'statistical'},
                                            {'label': 'Spectral', 'value': 'spectral'},
                                            {'label': 'Morphological', 'value': 'morphological'}
                                        ],
                                        value=['statistical', 'spectral', 'morphological']
                                    ),
                                    html.Br(),
                                    dbc.Button(
                                        "Extract Features",
                                        id="extract-features-button",
                                        color="primary",
                                        className="w-100"
                                    ),
                                    html.Div(id="feature-extraction-output")
                                ])
                            ], className="mb-4"),
                            
                            dbc.Card([
                                dbc.CardHeader("Model Training"),
                                dbc.CardBody([
                                    html.Label("Model Type"),
                                    dcc.Dropdown(
                                        id='model-type',
                                        options=[
                                            {'label': 'Classification', 'value': 'classification'},
                                            {'label': 'Clustering', 'value': 'clustering'},
                                            {'label': 'Anomaly Detection', 'value': 'anomaly_detection'}
                                        ],
                                        value='classification'
                                    ),
                                    html.Div(id='model-specific-controls'),
                                    html.Br(),
                                    dbc.Button(
                                        "Train Model",
                                        id="train-model-button",
                                        color="primary",
                                        className="w-100"
                                    ),
                                    html.Div(id="model-training-output")
                                ])
                            ], className="mb-4")
                        ], width=4),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("ML Results"),
                                dbc.CardBody([
                                    dcc.Graph(id='ml-results-graph'),
                                    html.Div(id='ml-results-table')
                                ])
                            ])
                        ], width=8)
                    ])
                ]
            )
        ]
