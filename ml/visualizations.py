import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import pandas as pd

class MLVisualizer:
    """
    Generate interactive visualizations for machine learning results on AFM data.
    
    This module provides methods for creating various visualizations:
    - PCA and t-SNE plots for dimensionality reduction
    - Cluster visualization
    - Feature importance plots
    - Anomaly detection visualization
    - Confusion matrices for classification results
    """
    
    def __init__(self, colorscale='Viridis'):
        """
        Initialize the visualizer.
        
        Args:
            colorscale (str): Default colorscale for plots
        """
        self.colorscale = colorscale
    
    def plot_dimensionality_reduction(self, features, labels=None, method='pca', 
                                     n_components=2, perplexity=30, title=None,
                                     animate=False, frame_column=None):
        """
        Create a dimensionality reduction plot using PCA or t-SNE.
        
        Args:
            features (numpy.ndarray): Feature array
            labels (numpy.ndarray, optional): Labels or cluster assignments
            method (str): Reduction method ('pca' or 'tsne')
            n_components (int): Number of components for the reduction
            perplexity (int): Perplexity parameter for t-SNE
            title (str, optional): Plot title
            animate (bool): Whether to create an animated plot (requires frame_column)
            frame_column (numpy.ndarray, optional): Values to use for animation frames
            
        Returns:
            plotly.graph_objects.Figure: Interactive plotly figure
        """
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
            reduced_data = reducer.fit_transform(features)
            explained_var = reducer.explained_variance_ratio_
            method_name = 'PCA'
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            reduced_data = reducer.fit_transform(features)
            explained_var = None
            method_name = 't-SNE'
        else:
            raise ValueError(f"Unsupported reduction method: {method}")
        
        # Create a pandas DataFrame for the plot
        if n_components == 2:
            df = pd.DataFrame({
                'x': reduced_data[:, 0],
                'y': reduced_data[:, 1]
            })
            
            # Add labels if provided
            if labels is not None:
                df['label'] = labels
                
                # Add animation frames if requested
                if animate and frame_column is not None:
                    df['frame'] = frame_column
                    
                    # Create animated plot
                    if title is None:
                        title = f"{method_name} Visualization of AFM Features (Animated)"
                    
                    fig = px.scatter(df, x='x', y='y', color='label', animation_frame='frame',
                                     title=title,
                                     labels={'x': f'{method_name} Component 1', 
                                             'y': f'{method_name} Component 2',
                                             'label': 'Label',
                                             'frame': 'Frame'},
                                     color_continuous_scale=self.colorscale)
                else:    
                    # Create static plot
                    if title is None:
                        title = f"{method_name} Visualization of AFM Features"
                    
                    fig = px.scatter(df, x='x', y='y', color='label',
                                     title=title,
                                     labels={'x': f'{method_name} Component 1', 
                                             'y': f'{method_name} Component 2',
                                             'label': 'Label'},
                                     color_continuous_scale=self.colorscale)
            else:
                # Add animation frames if requested
                if animate and frame_column is not None:
                    df['frame'] = frame_column
                    
                    # Create animated plot without color labels
                    if title is None:
                        title = f"{method_name} Visualization of AFM Features (Animated)"
                    
                    fig = px.scatter(df, x='x', y='y', animation_frame='frame',
                                     title=title,
                                     labels={'x': f'{method_name} Component 1', 
                                             'y': f'{method_name} Component 2',
                                             'frame': 'Frame'},
                                     color_continuous_scale=self.colorscale)
                else:
                    # Create static plot without color labels
                    if title is None:
                        title = f"{method_name} Visualization of AFM Features"
                    
                    fig = px.scatter(df, x='x', y='y',
                                     title=title,
                                     labels={'x': f'{method_name} Component 1', 
                                             'y': f'{method_name} Component 2'},
                                     color_continuous_scale=self.colorscale)
        
        elif n_components == 3:
            df = pd.DataFrame({
                'x': reduced_data[:, 0],
                'y': reduced_data[:, 1],
                'z': reduced_data[:, 2]
            })
            
            # Add labels if provided
            if labels is not None:
                df['label'] = labels
                
                # Create plot
                if title is None:
                    title = f"3D {method_name} Visualization of AFM Features"
                
                fig = px.scatter_3d(df, x='x', y='y', z='z', color='label',
                                   title=title,
                                   labels={'x': f'{method_name} Component 1', 
                                           'y': f'{method_name} Component 2',
                                           'z': f'{method_name} Component 3',
                                           'label': 'Label'},
                                   color_continuous_scale=self.colorscale)
            else:
                # Create plot without color labels
                if title is None:
                    title = f"3D {method_name} Visualization of AFM Features"
                
                fig = px.scatter_3d(df, x='x', y='y', z='z',
                                   title=title,
                                   labels={'x': f'{method_name} Component 1', 
                                           'y': f'{method_name} Component 2',
                                           'z': f'{method_name} Component 3'},
                                   color_continuous_scale=self.colorscale)
        else:
            raise ValueError(f"Can only visualize 2 or 3 components, got {n_components}")
        
        # Add explained variance if using PCA
        if explained_var is not None and n_components == 2:
            fig.update_layout(
                xaxis_title=f"{method_name} Component 1 ({explained_var[0]:.1%} variance)",
                yaxis_title=f"{method_name} Component 2 ({explained_var[1]:.1%} variance)"
            )
        elif explained_var is not None and n_components == 3:
            fig.update_layout(
                scene=dict(
                    xaxis_title=f"{method_name} Component 1 ({explained_var[0]:.1%} variance)",
                    yaxis_title=f"{method_name} Component 2 ({explained_var[1]:.1%} variance)",
                    zaxis_title=f"{method_name} Component 3 ({explained_var[2]:.1%} variance)"
                )
            )
        
        return fig
    
    def plot_pca_components(self, features, n_components=4, profile_shape=None, title="PCA Components"):
        """
        Visualize PCA components.
        
        Args:
            features (numpy.ndarray): Feature array
            n_components (int): Number of components to visualize
            profile_shape (tuple, optional): Shape of the original profiles (height, width)
                                           Not used in this simplified version
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive plotly figure with PCA components
        """
        try:
            # Apply PCA
            n_components = min(n_components, features.shape[1])  # Ensure we don't request more components than features
            pca = PCA(n_components=n_components)
            pca.fit(features)
            components = pca.components_
            explained_var = pca.explained_variance_ratio_
            
            # No reshaping - just use the components as 1D arrays
            # This avoids issues with trying to infer the correct 2D shape
        
            # Create subplots
            fig = make_subplots(
                rows=int(np.ceil(n_components/2)), cols=2,
                subplot_titles=[f"Component {i+1} ({explained_var[i]:.1%} variance)" 
                              for i in range(n_components)]
            )
            
            # Add line plots for each component instead of heatmaps
            for i in range(n_components):
                row = i // 2 + 1
                col = i % 2 + 1
                
                # Create a line plot of the component values
                x = np.arange(len(components[i]))
                fig.add_trace(
                    go.Scatter(
                        x=x, 
                        y=components[i],
                        mode='lines',
                        name=f"Component {i+1}",
                        line=dict(color=px.colors.sequential.Viridis[i % len(px.colors.sequential.Viridis)])
                    ),
                    row=row, col=col
                )
                
                # Update axes for this subplot
                fig.update_xaxes(title_text="Feature Index", row=row, col=col)
                fig.update_yaxes(title_text="Weight", row=row, col=col)
        
            # Update layout
            fig.update_layout(
                title=title,
                height=300 * int(np.ceil(n_components/2)),
                width=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            # Create a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error visualizing PCA components: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            fig.update_layout(title="PCA Components Visualization Error")
            return fig
    
    def plot_cluster_heatmap(self, profiles, cluster_labels, x_axis=None, y_axis=None, 
                           title="Cluster Assignment Heatmap"):
        """
        Create a heatmap showing the cluster assignments for AFM profiles.
        
        Args:
            profiles (numpy.ndarray): 3D array of profiles (depths, x, y)
            cluster_labels (numpy.ndarray): Cluster assignments
            x_axis (numpy.ndarray, optional): X-axis values
            y_axis (numpy.ndarray, optional): Y-axis values
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive plotly figure
        """
        # Reshape cluster labels to 2D grid if needed
        if profiles.ndim == 3:  # (depths, x, y)
            n_depths, n_x, n_y = profiles.shape
            reshaped_labels = cluster_labels.reshape(n_x, n_y)
        else:  # Assume profiles is (samples, features) and needs x, y coordinates
            if x_axis is None or y_axis is None:
                raise ValueError("Must provide x_axis and y_axis when profiles is not 3D")
            n_x = len(np.unique(x_axis))
            n_y = len(np.unique(y_axis))
            reshaped_labels = cluster_labels.reshape(n_x, n_y)
        
        # Create heatmap
        fig = px.imshow(
            reshaped_labels,
            title=title,
            color_continuous_scale=self.colorscale,
            labels={'color': 'Cluster'}
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="X Position",
            yaxis_title="Y Position"
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importance, feature_names=None, 
                              title="Feature Importance"):
        """
        Create a bar chart showing feature importance.
        
        Args:
            feature_importance (numpy.ndarray): Feature importance scores
            feature_names (list, optional): Names for each feature
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive plotly figure
        """
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(len(feature_importance))]
        
        # Create DataFrame for the plot
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })
        
        # Sort by importance
        df = df.sort_values('Importance', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            df,
            x='Feature',
            y='Importance',
            title=title,
            color='Importance',
            color_continuous_scale=self.colorscale
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Feature",
            yaxis_title="Importance Score",
            xaxis={'categoryorder': 'total descending'}
        )
        
        return fig
    
    def plot_anomaly_detection(self, profiles, anomaly_scores, anomalies=None, 
                             threshold=None, x_axis=None, y_axis=None,
                             title="Anomaly Detection"):
        """
        Create a visualization of anomaly detection results.
        
        Args:
            profiles (numpy.ndarray): 3D array of profiles (depths, x, y) or 2D array
            anomaly_scores (numpy.ndarray): Anomaly scores for each profile
            anomalies (numpy.ndarray, optional): Binary array indicating anomalies
            threshold (float, optional): Threshold used for anomaly detection
            x_axis (numpy.ndarray, optional): X-axis values (required if profiles is 2D)
            y_axis (numpy.ndarray, optional): Y-axis values (required if profiles is 2D)
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive plotly figure with two subplots:
                1. Heatmap of anomaly scores
                2. Binary map of anomalies (if provided)
        """
        # Reshape anomaly scores to 2D grid if needed
        if profiles.ndim == 3:  # (depths, x, y)
            n_depths, n_x, n_y = profiles.shape
            reshaped_scores = anomaly_scores.reshape(n_x, n_y)
            if anomalies is not None:
                reshaped_anomalies = anomalies.reshape(n_x, n_y)
        else:  # Assume profiles is (samples, features) and needs x, y coordinates
            if x_axis is None or y_axis is None:
                raise ValueError("Must provide x_axis and y_axis when profiles is not 3D")
            n_x = len(np.unique(x_axis))
            n_y = len(np.unique(y_axis))
            reshaped_scores = anomaly_scores.reshape(n_x, n_y)
            if anomalies is not None:
                reshaped_anomalies = anomalies.reshape(n_x, n_y)
        
        # Create figure with 1 or 2 subplots
        if anomalies is not None:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Anomaly Scores", "Detected Anomalies"],
                horizontal_spacing=0.1
            )
            
            # Add anomaly score heatmap
            fig.add_trace(
                go.Heatmap(
                    z=reshaped_scores,
                    colorscale=self.colorscale,
                    colorbar=dict(title="Score", x=0.46)
                ),
                row=1, col=1
            )
            
            # Add binary anomaly map
            fig.add_trace(
                go.Heatmap(
                    z=reshaped_anomalies,
                    colorscale=[[0, 'blue'], [1, 'red']],
                    showscale=False
                ),
                row=1, col=2
            )
            
            # Add threshold line to colorbar if provided
            if threshold is not None:
                fig.update_layout(
                    annotations=[
                        dict(
                            x=0.46,
                            y=threshold,
                            xref="paper",
                            yref="y",
                            text="Threshold",
                            showarrow=True,
                            arrowhead=2,
                            ax=40,
                            ay=0
                        )
                    ]
                )
        else:
            # Only anomaly score heatmap
            fig = go.Figure(
                data=go.Heatmap(
                    z=reshaped_scores,
                    colorscale=self.colorscale,
                    colorbar=dict(title="Anomaly Score")
                )
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=500,
            width=900
        )
        
        return fig
    
    def plot_pca_components(self, features, n_components=4, title=None, image_shape=None):
        """
        Visualize PCA components as images to show what features contribute to each component.
        
        Args:
            features (numpy.ndarray): Feature array
            n_components (int): Number of PCA components to visualize
            title (str, optional): Plot title
            image_shape (tuple, optional): Shape to reshape components into (default: square)
            
        Returns:
            plotly.graph_objects.Figure: Interactive plotly figure with PCA components
        """
        # Apply PCA to extract components
        pca = PCA(n_components=n_components)
        pca.fit(features)
        
        # Get components and explained variance
        components = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
        
        # Determine image shape if not provided
        if image_shape is None:
            # Try to make a square-ish shape based on feature dimensions
            feature_dim = features.shape[1]
            side_length = int(np.sqrt(feature_dim))
            image_shape = (side_length, feature_dim // side_length)
            
            # If perfect square not possible, adjust
            if side_length * (feature_dim // side_length) != feature_dim:
                # Just use a 1D representation
                image_shape = (1, feature_dim)
        
        # Create subplots - one for each component
        fig = make_subplots(
            rows=int(np.ceil(n_components/2)), 
            cols=2,
            subplot_titles=[f"Component {i+1} ({explained_variance_ratio[i]:.2%} variance)" 
                           for i in range(n_components)]
        )
        
        # Add each component as a heatmap
        for i in range(n_components):
            # Reshape component to image dimensions if possible
            try:
                component_img = components[i].reshape(image_shape)
            except ValueError:
                # If reshaping fails, keep as 1D
                component_img = components[i].reshape(1, -1)
                
            # Create heatmap for this component
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Heatmap(
                    z=component_img,
                    colorscale=self.colorscale,
                    showscale=i == 0,  # Only show colorbar for first component
                ),
                row=row, col=col
            )
            
            # Update axes
            fig.update_xaxes(title="Feature Dimension", row=row, col=col)
            fig.update_yaxes(title="Feature Dimension", row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title=title or f"PCA Components ({n_components} components)",
            height=300 * int(np.ceil(n_components/2)),
            width=800,
            showlegend=False
        )
        
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                            title="Confusion Matrix"):
        """
        Create a visualization of a confusion matrix for classification results.
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted labels
            class_names (list, optional): Names for each class
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive plotly figure
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create class names if not provided
        if class_names is None:
            class_names = [f"Class {i}" for i in range(cm.shape[0])]
        
        # Create heatmap
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale=self.colorscale,
            annotation_text=cm.astype(str),
            showscale=True
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(title="Predicted Label"),
            yaxis=dict(title="True Label", autorange="reversed")
        )
        
        return fig
    
    def plot_3d_clusters(self, x, y, z, cluster_labels, title="3D Cluster Visualization"):
        """
        Create a 3D scatter plot showing clusters.
        
        Args:
            x (numpy.ndarray): X-coordinates
            y (numpy.ndarray): Y-coordinates
            z (numpy.ndarray): Z-coordinates (e.g., depth values)
            cluster_labels (numpy.ndarray): Cluster assignments
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive 3D plotly figure
        """
        # Create a pandas DataFrame for the plot
        df = pd.DataFrame({
            'x': x,
            'y': y,
            'z': z,
            'cluster': cluster_labels
        })
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            color='cluster',
            title=title,
            color_continuous_scale=self.colorscale
        )
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Depth"
            )
        )
        
        return fig
    
    def plot_feature_distributions(self, features, labels=None, feature_names=None,
                                 title="Feature Distributions"):
        """
        Create a plot showing the distribution of feature values.
        
        Args:
            features (numpy.ndarray): Feature array
            labels (numpy.ndarray, optional): Labels or cluster assignments
            feature_names (list, optional): Names for each feature
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive plotly figure
        """
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(features.shape[1])]
        
        # Create a pandas DataFrame for the plot
        df = pd.DataFrame(features, columns=feature_names)
        
        if labels is not None:
            df['label'] = labels
            fig = px.box(df, y=feature_names, color='label', title=title)
        else:
            fig = px.box(df, y=feature_names, title=title)
        
        # Update layout
        fig.update_layout(
            xaxis_title="Value",
            yaxis_title="Feature"
        )
        
        return fig
