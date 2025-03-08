import numpy as np
import sklearn
from sklearn.decomposition import PCA, KernelPCA, NMF, FastICA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Class for unsupervised feature extraction from AFM depth profiles."""
    
    def __init__(self):
        self.models = {
            'pca': None,
            'kernel_pca': None,
            'nmf': None,
            'ica': None,
            'tsne': None,
            'isomap': None,
            'lle': None
        }
        self.available_methods = list(self.models.keys())
        self.fitted = False
        self.n_components = 2  # Default number of components
        
    def set_params(self, method, **kwargs):
        """Set parameters for the selected feature extraction method.
        
        Args:
            method (str): The feature extraction method to configure
            **kwargs: Parameters specific to the selected method
        """
        if method not in self.available_methods:
            raise ValueError(f"Method {method} not available. Choose from: {self.available_methods}")
            
        if method == 'pca':
            self.models[method] = PCA(**kwargs)
        elif method == 'kernel_pca':
            self.models[method] = KernelPCA(**kwargs)
        elif method == 'nmf':
            self.models[method] = NMF(**kwargs)
        elif method == 'ica':
            self.models[method] = FastICA(**kwargs)
        elif method == 'tsne':
            self.models[method] = TSNE(**kwargs)
        elif method == 'isomap':
            self.models[method] = Isomap(**kwargs)
        elif method == 'lle':
            self.models[method] = LocallyLinearEmbedding(**kwargs)
            
        logger.info(f"Configured {method} with parameters: {kwargs}")
        
    def fit(self, data, method, n_components=2):
        """Fit the feature extraction model to the data.
        
        Args:
            data (numpy.ndarray): 2D array where each row is a depth profile
            method (str): Feature extraction method to use
            n_components (int): Number of components/features to extract
            
        Returns:
            self: The fitted extractor instance
        """
        if method not in self.available_methods:
            raise ValueError(f"Method {method} not available. Choose from: {self.available_methods}")
        
        self.n_components = n_components
        
        # Prepare data - handle NaN values
        data_clean = np.nan_to_num(data)
        
        # Configure the model if not already configured
        if self.models[method] is None:
            self.set_params(method, n_components=n_components)
        
        # Fit the model
        logger.info(f"Fitting {method} with {n_components} components on data shape: {data_clean.shape}")
        try:
            self.models[method].fit(data_clean)
            self.fitted = True
            logger.info(f"Successfully fitted {method}")
        except Exception as e:
            logger.error(f"Error fitting {method}: {str(e)}")
            raise
            
        return self
    
    def transform(self, data, method):
        """Transform data using the fitted feature extraction model.
        
        Args:
            data (numpy.ndarray): 2D array where each row is a depth profile
            method (str): Feature extraction method to use
            
        Returns:
            numpy.ndarray: Transformed data with reduced dimensions
        """
        if not self.fitted or self.models[method] is None:
            raise ValueError(f"Model {method} not fitted yet. Call fit() first.")
        
        # Prepare data - handle NaN values
        data_clean = np.nan_to_num(data)
        
        logger.info(f"Transforming data with shape {data_clean.shape} using {method}")
        try:
            transformed_data = self.models[method].transform(data_clean)
            logger.info(f"Successfully transformed data to shape {transformed_data.shape}")
            return transformed_data
        except Exception as e:
            logger.error(f"Error transforming with {method}: {str(e)}")
            raise
    
    def fit_transform(self, data, method, n_components=2):
        """Fit the model and transform the data in one step.
        
        Args:
            data (numpy.ndarray): 2D array where each row is a depth profile
            method (str): Feature extraction method to use
            n_components (int): Number of components/features to extract
            
        Returns:
            numpy.ndarray: Transformed data with reduced dimensions
        """
        self.fit(data, method, n_components)
        return self.transform(data, method)
    
    def get_components(self, method):
        """Get the components/directions learned by the model.
        
        Args:
            method (str): Feature extraction method
            
        Returns:
            numpy.ndarray: Components matrix if available, None otherwise
        """
        if not self.fitted or self.models[method] is None:
            raise ValueError(f"Model {method} not fitted yet. Call fit() first.")
        
        # Only some models have components that can be retrieved
        if method == 'pca':
            return self.models[method].components_
        elif method == 'nmf':
            return self.models[method].components_
        elif method == 'ica':
            return self.models[method].components_
        else:
            logger.warning(f"Components not available for method {method}")
            return None
    
    def get_explained_variance(self, method):
        """Get explained variance ratio for methods that support it (e.g., PCA).
        
        Args:
            method (str): Feature extraction method
            
        Returns:
            numpy.ndarray: Explained variance ratio if available, None otherwise
        """
        if not self.fitted or self.models[method] is None:
            raise ValueError(f"Model {method} not fitted yet. Call fit() first.")
        
        if method == 'pca':
            return self.models[method].explained_variance_ratio_
        else:
            logger.warning(f"Explained variance not available for method {method}")
            return None


class ClusteringAnalysis:
    """Class for unsupervised clustering of AFM depth profiles."""
    
    def __init__(self):
        self.models = {
            'kmeans': None,
            'dbscan': None,
            'spectral': None
        }
        self.available_methods = list(self.models.keys())
        self.fitted = False
        
    def set_params(self, method, **kwargs):
        """Set parameters for the selected clustering method.
        
        Args:
            method (str): The clustering method to configure
            **kwargs: Parameters specific to the selected method
        """
        if method not in self.available_methods:
            raise ValueError(f"Method {method} not available. Choose from: {self.available_methods}")
            
        if method == 'kmeans':
            self.models[method] = KMeans(**kwargs)
        elif method == 'dbscan':
            self.models[method] = DBSCAN(**kwargs)
        elif method == 'spectral':
            self.models[method] = SpectralClustering(**kwargs)
            
        logger.info(f"Configured {method} with parameters: {kwargs}")
        
    def fit(self, data, method, **kwargs):
        """Fit the clustering model to the data.
        
        Args:
            data (numpy.ndarray): 2D array where each row is a depth profile or extracted features
            method (str): Clustering method to use
            **kwargs: Additional parameters for the clustering method
            
        Returns:
            self: The fitted clustering instance
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
            self.models[method].fit(data_clean)
            self.fitted = True
            logger.info(f"Successfully fitted {method}")
        except Exception as e:
            logger.error(f"Error fitting {method}: {str(e)}")
            raise
            
        return self
    
    def predict(self, data, method):
        """Predict cluster labels for new data.
        
        Args:
            data (numpy.ndarray): 2D array where each row is a depth profile or extracted features
            method (str): Clustering method to use
            
        Returns:
            numpy.ndarray: Cluster labels for each sample
        """
        if not self.fitted or self.models[method] is None:
            raise ValueError(f"Model {method} not fitted yet. Call fit() first.")
        
        # Prepare data - handle NaN values
        data_clean = np.nan_to_num(data)
        
        logger.info(f"Predicting clusters for data with shape {data_clean.shape} using {method}")
        
        # For DBSCAN, we need to handle prediction differently
        if method == 'dbscan':
            # DBSCAN doesn't have a predict method, so we return the labels from fit
            # This is only valid if the data is the same as used in fit
            return self.models[method].labels_
        
        try:
            labels = self.models[method].predict(data_clean)
            logger.info(f"Successfully predicted clusters: {np.unique(labels)}")
            return labels
        except Exception as e:
            logger.error(f"Error predicting with {method}: {str(e)}")
            raise
    
    def fit_predict(self, data, method, **kwargs):
        """Fit the model and predict cluster labels in one step.
        
        Args:
            data (numpy.ndarray): 2D array where each row is a depth profile or extracted features
            method (str): Clustering method to use
            **kwargs: Additional parameters for the clustering method
            
        Returns:
            numpy.ndarray: Cluster labels for each sample
        """
        self.fit(data, method, **kwargs)
        return self.predict(data, method)
    
    def get_cluster_centers(self, method):
        """Get cluster centers for methods that support it (e.g., KMeans).
        
        Args:
            method (str): Clustering method
            
        Returns:
            numpy.ndarray: Cluster centers if available, None otherwise
        """
        if not self.fitted or self.models[method] is None:
            raise ValueError(f"Model {method} not fitted yet. Call fit() first.")
        
        if method == 'kmeans':
            return self.models[method].cluster_centers_
        else:
            logger.warning(f"Cluster centers not available for method {method}")
            return None
