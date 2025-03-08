import numpy as np
import pickle
import os

class ModelInference:
    """
    Apply trained machine learning models to new AFM depth profile data.
    
    This module provides methods for:
    - Loading trained models
    - Applying classification models to identify materials
    - Applying clustering models to segment profiles
    - Applying anomaly detection models to identify defects
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the inference module.
        
        Args:
            model_path (str, optional): Path to a pre-trained model file
        """
        self.model_data = None
        if model_path is not None:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load a trained model from file.
        
        Args:
            model_path (str): Path to the saved model file
            
        Returns:
            bool: True if model was loaded successfully
        """
        try:
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _preprocess_features(self, features):
        """
        Preprocess features using the scaler from the loaded model.
        
        Args:
            features (numpy.ndarray): Feature array
            
        Returns:
            numpy.ndarray: Scaled features
        """
        if self.model_data is None or 'scaler' not in self.model_data:
            raise ValueError("No model loaded or model does not contain a scaler")
        
        # Validate feature dimensions
        if 'feature_config' in self.model_data:
            expected_features = self.model_data['feature_config'].get('n_features', None)
            if expected_features is not None and features.shape[1] != expected_features:
                raise ValueError(
                    f"Expected {expected_features} features, got {features.shape[1]}. "
                    "Ensure the same feature extraction configuration is used."
                )
        
        return self.model_data['scaler'].transform(features)
    
    def classify(self, features, return_probabilities=False):
        """
        Apply a trained classification model to identify materials.
        
        Args:
            features (numpy.ndarray): Feature array
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            tuple: (predictions, probabilities) if return_probabilities is True,
                  otherwise just predictions
        """
        if self.model_data is None or 'model' not in self.model_data:
            raise ValueError("No model loaded")
        
        # Check if this is a classification model
        if not hasattr(self.model_data['model'], 'predict_proba'):
            raise ValueError("Loaded model is not a classifier")
        
        # Preprocess features
        features_scaled = self._preprocess_features(features)
        
        # Make predictions
        predictions_encoded = self.model_data['model'].predict(features_scaled)
        
        # Convert encoded predictions back to original labels
        if 'label_encoder' in self.model_data:
            predictions = self.model_data['label_encoder'].inverse_transform(predictions_encoded)
        else:
            predictions = predictions_encoded
        
        if return_probabilities:
            probabilities = self.model_data['model'].predict_proba(features_scaled)
            return predictions, probabilities
        else:
            return predictions
    
    def cluster(self, features):
        """
        Apply a trained clustering model to segment profiles.
        
        Args:
            features (numpy.ndarray): Feature array
            
        Returns:
            numpy.ndarray: Cluster assignments
        """
        if self.model_data is None or 'model' not in self.model_data:
            raise ValueError("No model loaded")
        
        # Check if this is a clustering model
        if not hasattr(self.model_data['model'], 'predict') and not hasattr(self.model_data['model'], 'fit_predict'):
            raise ValueError("Loaded model is not a clustering model")
        
        # Preprocess features
        features_scaled = self._preprocess_features(features)
        
        # Make predictions
        if hasattr(self.model_data['model'], 'predict'):
            cluster_assignments = self.model_data['model'].predict(features_scaled)
        else:
            # For models like DBSCAN that only have fit_predict
            # This is not ideal as it will refit the model, but it's a fallback
            cluster_assignments = self.model_data['model'].fit_predict(features_scaled)
        
        return cluster_assignments
    
    def detect_anomalies(self, features, threshold=None):
        """
        Apply a trained anomaly detection model to identify defects.
        
        Args:
            features (numpy.ndarray): Feature array
            threshold (float, optional): Custom threshold for anomaly detection
                (if None, use the model's default threshold)
            
        Returns:
            tuple: (anomalies, anomaly_scores)
                - anomalies: Binary array (1 for anomalies, 0 for normal)
                - anomaly_scores: Anomaly scores (higher = more anomalous)
        """
        if self.model_data is None or 'model' not in self.model_data:
            raise ValueError("No model loaded")
        
        # Check if this is an anomaly detection model
        if not hasattr(self.model_data['model'], 'predict') or not hasattr(self.model_data['model'], 'score_samples'):
            raise ValueError("Loaded model is not an anomaly detection model")
        
        # Preprocess features
        features_scaled = self._preprocess_features(features)
        
        # Get anomaly scores
        anomaly_scores = -self.model_data['model'].score_samples(features_scaled)  # Negate so higher = more anomalous
        
        # Get anomaly predictions
        if threshold is None:
            # Use model's default threshold
            predictions = self.model_data['model'].predict(features_scaled)
            anomalies = np.where(predictions == -1, 1, 0)  # Convert to 1 for anomalies, 0 for normal
        else:
            # Use custom threshold
            anomalies = np.where(anomaly_scores > threshold, 1, 0)
        
        return anomalies, anomaly_scores
    
    def get_feature_importance(self):
        """
        Get feature importance from the loaded model if available.
        
        Returns:
            numpy.ndarray or None: Feature importance scores if available
        """
        if self.model_data is None:
            raise ValueError("No model loaded")
        
        if 'feature_importances_' in self.model_data:
            return self.model_data['feature_importances_']
        elif hasattr(self.model_data['model'], 'feature_importances_'):
            return self.model_data['model'].feature_importances_
        else:
            return None
    
    def get_model_metrics(self):
        """
        Get metrics of the loaded model.
        
        Returns:
            dict or None: Model metrics if available
        """
        if self.model_data is None:
            raise ValueError("No model loaded")
        
        return self.model_data.get('metrics', None)
    
    def get_model_type(self):
        """
        Determine the type of the loaded model.
        
        Returns:
            str: Type of model ('classifier', 'clustering', or 'anomaly_detector')
        """
        if self.model_data is None or 'model' not in self.model_data:
            raise ValueError("No model loaded")
        
        model = self.model_data['model']
        
        if hasattr(model, 'predict_proba'):
            return 'classifier'
        elif hasattr(model, 'score_samples') and hasattr(model, 'predict'):
            # Isolation Forest and other anomaly detectors typically have these methods
            return 'anomaly_detector'
        elif hasattr(model, 'predict') or hasattr(model, 'fit_predict'):
            # Most clustering algorithms have at least one of these
            return 'clustering'
        else:
            return 'unknown'
