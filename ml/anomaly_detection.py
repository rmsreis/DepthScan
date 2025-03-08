import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
import logging

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Class for unsupervised anomaly detection in AFM depth profiles."""
    
    def __init__(self):
        self.models = {
            'isolation_forest': None,
            'local_outlier_factor': None,
            'one_class_svm': None,
            'robust_covariance': None
        }
        self.available_methods = list(self.models.keys())
        self.fitted = False
        
    def set_params(self, method, **kwargs):
        """Set parameters for the selected anomaly detection method.
        
        Args:
            method (str): The anomaly detection method to configure
            **kwargs: Parameters specific to the selected method
        """
        if method not in self.available_methods:
            raise ValueError(f"Method {method} not available. Choose from: {self.available_methods}")
            
        if method == 'isolation_forest':
            self.models[method] = IsolationForest(**kwargs)
        elif method == 'local_outlier_factor':
            self.models[method] = LocalOutlierFactor(**kwargs)
        elif method == 'one_class_svm':
            self.models[method] = OneClassSVM(**kwargs)
        elif method == 'robust_covariance':
            self.models[method] = EllipticEnvelope(**kwargs)
            
        logger.info(f"Configured {method} with parameters: {kwargs}")
        
    def fit(self, data, method, **kwargs):
        """Fit the anomaly detection model to the data.
        
        Args:
            data (numpy.ndarray): 2D array where each row is a depth profile or extracted features
            method (str): Anomaly detection method to use
            **kwargs: Additional parameters for the anomaly detection method
            
        Returns:
            self: The fitted detector instance
        """
        if method not in self.available_methods:
            raise ValueError(f"Method {method} not available. Choose from: {self.available_methods}")
        
        # Configure the model with the provided parameters
        self.set_params(method, **kwargs)
        
        # Prepare data - handle NaN values
        data_clean = np.nan_to_num(data)
        
        # Fit the model
        logger.info(f"Fitting {method} on data shape: {data_clean.shape}")
        try:
            # LocalOutlierFactor in novelty detection mode needs fit_predict
            if method == 'local_outlier_factor':
                self.models[method].fit_predict(data_clean)
            else:
                self.models[method].fit(data_clean)
            self.fitted = True
            logger.info(f"Successfully fitted {method}")
        except Exception as e:
            logger.error(f"Error fitting {method}: {str(e)}")
            raise
            
        return self
    
    def predict(self, data, method, return_scores=False):
        """Predict anomalies in new data.
        
        Args:
            data (numpy.ndarray): 2D array where each row is a depth profile or extracted features
            method (str): Anomaly detection method to use
            return_scores (bool): If True, return anomaly scores instead of binary labels
            
        Returns:
            numpy.ndarray: Anomaly labels (-1 for anomalies, 1 for normal) or scores
        """
        if not self.fitted or self.models[method] is None:
            raise ValueError(f"Model {method} not fitted yet. Call fit() first.")
        
        # Prepare data - handle NaN values
        data_clean = np.nan_to_num(data)
        
        logger.info(f"Predicting anomalies for data with shape {data_clean.shape} using {method}")
        
        try:
            if return_scores:
                # For methods that support decision_function for scoring
                if method in ['isolation_forest', 'one_class_svm', 'robust_covariance']:
                    scores = self.models[method].decision_function(data_clean)
                    logger.info(f"Successfully computed anomaly scores")
                    return scores
                # For LOF, use negative_outlier_factor_
                elif method == 'local_outlier_factor':
                    # Need to fit again on the new data to get scores
                    lof = LocalOutlierFactor(novelty=False, **self.models[method].get_params())
                    lof.fit(data_clean)
                    scores = -lof.negative_outlier_factor_
                    logger.info(f"Successfully computed anomaly scores")
                    return scores
            else:
                # Binary prediction (-1 for anomalies, 1 for normal)
                if method == 'local_outlier_factor' and not hasattr(self.models[method], 'predict'):
                    # For LOF in non-novelty mode
                    lof = LocalOutlierFactor(novelty=False, **self.models[method].get_params())
                    labels = lof.fit_predict(data_clean)
                else:
                    labels = self.models[method].predict(data_clean)
                logger.info(f"Successfully predicted anomalies: {np.sum(labels == -1)} found")
                return labels
        except Exception as e:
            logger.error(f"Error predicting with {method}: {str(e)}")
            raise
    
    def fit_predict(self, data, method, **kwargs):
        """Fit the model and predict anomalies in one step.
        
        Args:
            data (numpy.ndarray): 2D array where each row is a depth profile or extracted features
            method (str): Anomaly detection method to use
            **kwargs: Additional parameters for the anomaly detection method
            
        Returns:
            numpy.ndarray: Anomaly labels (-1 for anomalies, 1 for normal)
        """
        self.fit(data, method, **kwargs)
        return self.predict(data, method)
    
    def get_threshold(self, method, contamination=0.1):
        """Get the decision threshold for anomaly detection.
        
        Args:
            method (str): Anomaly detection method
            contamination (float): Expected proportion of outliers in the data
            
        Returns:
            float: Decision threshold if available, None otherwise
        """
        if not self.fitted or self.models[method] is None:
            raise ValueError(f"Model {method} not fitted yet. Call fit() first.")
        
        if method in ['isolation_forest', 'robust_covariance']:
            return self.models[method].threshold_
        else:
            logger.warning(f"Threshold not directly available for method {method}")
            return None


class AnomalyAnalyzer:
    """Class for analyzing and interpreting anomalies in AFM depth profiles."""
    
    def __init__(self, detector):
        """Initialize the analyzer with a fitted anomaly detector.
        
        Args:
            detector (AnomalyDetector): A fitted anomaly detector instance
        """
        self.detector = detector
        
    def get_anomaly_statistics(self, data, method, labels=None):
        """Get statistics about detected anomalies.
        
        Args:
            data (numpy.ndarray): 2D array where each row is a depth profile
            method (str): Anomaly detection method used
            labels (numpy.ndarray, optional): Pre-computed anomaly labels
            
        Returns:
            dict: Statistics about the anomalies
        """
        if labels is None:
            labels = self.detector.predict(data, method)
        
        anomaly_indices = np.where(labels == -1)[0]
        normal_indices = np.where(labels == 1)[0]
        
        stats = {
            'total_samples': len(data),
            'anomaly_count': len(anomaly_indices),
            'normal_count': len(normal_indices),
            'anomaly_percentage': 100 * len(anomaly_indices) / len(data),
            'anomaly_indices': anomaly_indices.tolist()
        }
        
        # Calculate statistics of anomalies vs normal samples
        if len(anomaly_indices) > 0 and len(normal_indices) > 0:
            anomaly_data = data[anomaly_indices]
            normal_data = data[normal_indices]
            
            stats['anomaly_mean'] = np.mean(anomaly_data, axis=0).tolist()
            stats['normal_mean'] = np.mean(normal_data, axis=0).tolist()
            stats['anomaly_std'] = np.std(anomaly_data, axis=0).tolist()
            stats['normal_std'] = np.std(normal_data, axis=0).tolist()
            
            # Calculate the features that differ most between anomalies and normal samples
            mean_diff = np.abs(stats['anomaly_mean'] - stats['normal_mean'])
            top_diff_indices = np.argsort(mean_diff)[-5:]  # Top 5 different features
            stats['top_different_features'] = top_diff_indices.tolist()
            stats['feature_difference_magnitude'] = mean_diff[top_diff_indices].tolist()
        
        return stats
    
    def find_optimal_contamination(self, data, method, contamination_range=None):
        """Find the optimal contamination parameter for anomaly detection.
        
        Args:
            data (numpy.ndarray): 2D array where each row is a depth profile
            method (str): Anomaly detection method to use
            contamination_range (list, optional): Range of contamination values to try
            
        Returns:
            dict: Results of the contamination analysis
        """
        if contamination_range is None:
            contamination_range = [0.01, 0.05, 0.1, 0.15, 0.2]
        
        results = []
        for contamination in contamination_range:
            # Configure and fit the model with this contamination
            if method == 'isolation_forest':
                self.detector.set_params(method, contamination=contamination, random_state=42)
            elif method == 'robust_covariance':
                self.detector.set_params(method, contamination=contamination, random_state=42)
            elif method == 'one_class_svm':
                # OneClassSVM doesn't have a contamination parameter, use nu instead
                self.detector.set_params(method, nu=contamination, gamma='auto')
            else:
                logger.warning(f"Method {method} doesn't support contamination parameter")
                continue
                
            # Fit and predict
            self.detector.fit(data, method)
            labels = self.detector.predict(data, method)
            
            # Get statistics
            stats = self.get_anomaly_statistics(data, method, labels)
            stats['contamination'] = contamination
            results.append(stats)
        
        return results
    
    def compare_methods(self, data, methods=None):
        """Compare different anomaly detection methods on the same data.
        
        Args:
            data (numpy.ndarray): 2D array where each row is a depth profile
            methods (list, optional): List of methods to compare
            
        Returns:
            dict: Comparison results for each method
        """
        if methods is None:
            methods = self.detector.available_methods
        
        results = {}
        for method in methods:
            try:
                # Configure with default parameters
                if method == 'isolation_forest':
                    self.detector.set_params(method, contamination='auto', random_state=42)
                elif method == 'local_outlier_factor':
                    self.detector.set_params(method, n_neighbors=20, novelty=True)
                elif method == 'one_class_svm':
                    self.detector.set_params(method, nu=0.1, gamma='auto')
                elif method == 'robust_covariance':
                    self.detector.set_params(method, contamination=0.1, random_state=42)
                
                # Fit and predict
                self.detector.fit(data, method)
                labels = self.detector.predict(data, method)
                
                # Get statistics
                stats = self.get_anomaly_statistics(data, method, labels)
                results[method] = stats
            except Exception as e:
                logger.error(f"Error comparing method {method}: {str(e)}")
                results[method] = {'error': str(e)}
        
        return results
