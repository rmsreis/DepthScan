import numpy as np
from scipy import stats, signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FeatureExtractor:
    """
    Extract meaningful features from AFM depth profiles for machine learning analysis.
    
    This module provides various feature extraction methods for AFM data, including:
    - Statistical features (mean, variance, skewness, etc.)
    - Spectral features (FFT components, power spectrum)
    - Morphological features (peaks, valleys, roughness)
    - Dimensionality reduction (PCA)
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.fitted = False
        self.n_features = 0
        self.statistical_enabled = True
        self.spectral_enabled = True
        self.morphological_enabled = True
    
    def extract_statistical_features(self, profiles):
        """
        Extract statistical features from AFM profiles.
        
        Args:
            profiles (numpy.ndarray): 2D array of profiles with shape (n_profiles, profile_length)
                or 3D array with shape (n_depths, n_profiles, profile_length)
                
        Returns:
            numpy.ndarray: Array of statistical features for each profile
        """
        # Handle different input dimensions
        if profiles.ndim == 3:
            # Reshape to 2D: (n_depths * n_profiles, profile_length)
            n_depths, n_profiles, profile_length = profiles.shape
            profiles_2d = profiles.reshape(-1, profile_length)
        else:
            profiles_2d = profiles
        
        # Initialize feature array
        n_samples = profiles_2d.shape[0]
        features = np.zeros((n_samples, 10))
        
        for i, profile in enumerate(profiles_2d):
            # Basic statistics
            features[i, 0] = np.mean(profile)                  # Mean
            features[i, 1] = np.std(profile)                   # Standard deviation
            features[i, 2] = stats.skew(profile)               # Skewness
            features[i, 3] = stats.kurtosis(profile)           # Kurtosis
            
            # Range statistics
            features[i, 4] = np.max(profile) - np.min(profile) # Range
            features[i, 5] = np.percentile(profile, 75) - np.percentile(profile, 25)  # IQR
            
            # Gradient statistics
            gradient = np.gradient(profile)
            features[i, 6] = np.mean(np.abs(gradient))         # Mean absolute gradient
            features[i, 7] = np.std(gradient)                  # Gradient standard deviation
            
            # Peak statistics
            peaks, _ = signal.find_peaks(profile)
            if len(peaks) > 0:
                features[i, 8] = len(peaks)                    # Number of peaks
                features[i, 9] = np.mean(profile[peaks])       # Mean peak height
            else:
                features[i, 8] = 0
                features[i, 9] = 0
        
        # Reshape back to include depth dimension if needed
        if profiles.ndim == 3:
            features = features.reshape(n_depths, n_profiles, -1)
        
        return features
    
    def extract_spectral_features(self, profiles, n_components=10):
        """
        Extract spectral features from AFM profiles using FFT.
        
        Args:
            profiles (numpy.ndarray): 2D array of profiles with shape (n_profiles, profile_length)
                or 3D array with shape (n_depths, n_profiles, profile_length)
            n_components (int): Number of frequency components to extract
                
        Returns:
            numpy.ndarray: Array of spectral features for each profile
        """
        # Handle different input dimensions
        if profiles.ndim == 3:
            n_depths, n_profiles, profile_length = profiles.shape
            profiles_2d = profiles.reshape(-1, profile_length)
        else:
            profiles_2d = profiles
        
        n_samples = profiles_2d.shape[0]
        features = np.zeros((n_samples, n_components))
        
        for i, profile in enumerate(profiles_2d):
            # Apply FFT
            fft_values = np.abs(np.fft.rfft(profile))
            
            # Get the most significant frequency components
            if len(fft_values) >= n_components:
                # Take the first n_components (lower frequencies often more important)
                features[i, :] = fft_values[:n_components]
            else:
                # Pad with zeros if not enough components
                features[i, :len(fft_values)] = fft_values
        
        # Reshape back to include depth dimension if needed
        if profiles.ndim == 3:
            features = features.reshape(n_depths, n_profiles, -1)
        
        return features
    
    def extract_morphological_features(self, profiles):
        """
        Extract morphological features from AFM profiles.
        
        Args:
            profiles (numpy.ndarray): 2D array of profiles with shape (n_profiles, profile_length)
                or 3D array with shape (n_depths, n_profiles, profile_length)
                
        Returns:
            numpy.ndarray: Array of morphological features for each profile
        """
        # Handle different input dimensions
        if profiles.ndim == 3:
            n_depths, n_profiles, profile_length = profiles.shape
            profiles_2d = profiles.reshape(-1, profile_length)
        else:
            profiles_2d = profiles
        
        n_samples = profiles_2d.shape[0]
        features = np.zeros((n_samples, 8))
        
        for i, profile in enumerate(profiles_2d):
            # Find peaks and valleys
            peaks, _ = signal.find_peaks(profile)
            valleys, _ = signal.find_peaks(-profile)
            
            # Calculate roughness (Ra - average roughness)
            mean_height = np.mean(profile)
            roughness = np.mean(np.abs(profile - mean_height))
            
            # Calculate peak-to-valley height (Rt)
            if len(peaks) > 0 and len(valleys) > 0:
                peak_to_valley = np.max(profile[peaks]) - np.min(profile[valleys])
            else:
                peak_to_valley = np.max(profile) - np.min(profile)
            
            # Calculate average peak spacing
            if len(peaks) > 1:
                peak_spacing = np.mean(np.diff(peaks))
            else:
                peak_spacing = 0
            
            # Calculate RMS roughness (Rq)
            rms_roughness = np.sqrt(np.mean((profile - mean_height) ** 2))
            
            # Calculate skewness of the height distribution (Rsk)
            height_skewness = stats.skew(profile)
            
            # Calculate kurtosis of the height distribution (Rku)
            height_kurtosis = stats.kurtosis(profile)
            
            # Store features
            features[i, 0] = len(peaks)       # Number of peaks
            features[i, 1] = len(valleys)     # Number of valleys
            features[i, 2] = roughness        # Average roughness (Ra)
            features[i, 3] = peak_to_valley   # Peak-to-valley height (Rt)
            features[i, 4] = peak_spacing     # Average peak spacing
            features[i, 5] = rms_roughness    # RMS roughness (Rq)
            features[i, 6] = height_skewness  # Skewness of height distribution (Rsk)
            features[i, 7] = height_kurtosis  # Kurtosis of height distribution (Rku)
        
        # Reshape back to include depth dimension if needed
        if profiles.ndim == 3:
            features = features.reshape(n_depths, n_profiles, -1)
        
        return features
    
    def apply_pca(self, features, n_components=0.95, fit=False):
        """
        Apply PCA dimensionality reduction to features.
        
        Args:
            features (numpy.ndarray): Feature array
            n_components (int or float): Number of components to keep (if int)
                or variance to preserve (if float between 0 and 1)
            fit (bool): Whether to fit a new PCA model or use an existing one
                
        Returns:
            numpy.ndarray: PCA-transformed features
        """
        # Reshape to 2D if needed
        original_shape = features.shape
        if features.ndim > 2:
            features_2d = features.reshape(-1, features.shape[-1])
        else:
            features_2d = features
        
        # Standardize features
        if fit or not self.fitted:
            features_scaled = self.scaler.fit_transform(features_2d)
            self.pca = PCA(n_components=n_components)
            features_pca = self.pca.fit_transform(features_scaled)
            self.fitted = True
        else:
            features_scaled = self.scaler.transform(features_2d)
            features_pca = self.pca.transform(features_scaled)
        
        # Reshape back if needed
        if features.ndim > 2:
            new_shape = list(original_shape[:-1]) + [-1]
            features_pca = features_pca.reshape(new_shape)
        
        return features_pca
    
    def extract_all_features(self, profiles, apply_pca=True, n_components=0.95):
        """
        Extract all available features and optionally apply PCA.
        
        Args:
            profiles (numpy.ndarray): 2D array of profiles with shape (n_profiles, profile_length)
                or 3D array with shape (n_depths, n_profiles, profile_length)
            apply_pca (bool): Whether to apply PCA to reduce dimensionality
            n_components (int or float): Number of components to keep (if int)
                or variance to preserve (if float between 0 and 1)
                
        Returns:
            numpy.ndarray: Combined feature array, optionally PCA-transformed
        """
        # Extract all types of features
        statistical_features = self.extract_statistical_features(profiles)
        spectral_features = self.extract_spectral_features(profiles)
        morphological_features = self.extract_morphological_features(profiles)
        
        # Combine features
        if profiles.ndim == 3:
            # For 3D input, concatenate along the last dimension
            combined_features = np.concatenate(
                [statistical_features, spectral_features, morphological_features],
                axis=2
            )
        else:
            # For 2D input, concatenate along the second dimension
            combined_features = np.concatenate(
                [statistical_features, spectral_features, morphological_features],
                axis=1
            )
        
        # Apply PCA if requested
        if apply_pca:
            return self.apply_pca(combined_features, n_components=n_components, fit=True)
        else:
            return combined_features
    
    def get_config(self):
        """
        Get the current feature extraction configuration.
        
        Returns:
            dict: Configuration including number of features and extraction methods
        """
        return {
            'n_features': self.n_features,
            'statistical': self.statistical_enabled,
            'spectral': self.spectral_enabled,
            'morphological': self.morphological_enabled
        }
