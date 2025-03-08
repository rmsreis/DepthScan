import os
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

class ModelTrainer:
    """
    Interface for training machine learning models on AFM depth profile features.
    
    This module provides methods for training various types of models:
    - Classification models (SVM, Random Forest) for material identification
    - Clustering models (K-means, DBSCAN) for unsupervised material segmentation
    - Anomaly detection models (Isolation Forest) for defect identification
    """
    
    def __init__(self, model_dir='ml/models'):
        """
        Initialize the model trainer.
        
        Args:
            model_dir (str): Directory to save trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def _preprocess_data(self, X, y=None, fit=True):
        """
        Preprocess the data by scaling features and encoding labels.
        
        Args:
            X (numpy.ndarray): Feature array
            y (numpy.ndarray, optional): Label array
            fit (bool): Whether to fit the scaler and encoder or just transform
            
        Returns:
            tuple: (X_scaled, y_encoded) or X_scaled if y is None
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        if y is not None:
            if fit:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
        else:
            return X_scaled
    
    def train_classifier(self, X, y, model_type='random_forest', test_size=0.2, 
                         random_state=42, hyperparams=None):
        """
        Train a classification model for material identification.
        
        Args:
            X (numpy.ndarray): Feature array
            y (numpy.ndarray): Label array (material types)
            model_type (str): Type of classifier ('random_forest' or 'svm')
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            hyperparams (dict, optional): Hyperparameters for the model
            
        Returns:
            dict: Dictionary containing the trained model, metrics, and preprocessing objects
        """
        # Preprocess data
        X_scaled, y_encoded = self._preprocess_data(X, y)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Initialize model based on type
        if model_type == 'random_forest':
            if hyperparams is None:
                hyperparams = {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                }
            model = RandomForestClassifier(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', None),
                min_samples_split=hyperparams.get('min_samples_split', 2),
                min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
                random_state=random_state
            )
        elif model_type == 'svm':
            if hyperparams is None:
                hyperparams = {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale'
                }
            model = SVC(
                C=hyperparams.get('C', 1.0),
                kernel=hyperparams.get('kernel', 'rbf'),
                gamma=hyperparams.get('gamma', 'scale'),
                probability=True,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Save the model
        timestamp = int(time.time())
        model_filename = f"{model_type}_classifier_{timestamp}.pkl"
        model_path = os.path.join(self.model_dir, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'metrics': metrics,
                'feature_importances': getattr(model, 'feature_importances_', None),
                'classes': self.label_encoder.classes_.tolist()
            }, f)
        
        return {
            'model': model,
            'metrics': metrics,
            'model_path': model_path,
            'classes': self.label_encoder.classes_.tolist(),
            'feature_importances': getattr(model, 'feature_importances_', None)
        }
    
    def train_clustering(self, X, n_clusters=None, model_type='kmeans', eps=0.5, 
                        min_samples=5, random_state=42):
        """
        Train a clustering model for unsupervised material segmentation.
        
        Args:
            X (numpy.ndarray): Feature array
            n_clusters (int, optional): Number of clusters for K-means
            model_type (str): Type of clustering ('kmeans' or 'dbscan')
            eps (float): Maximum distance between samples for DBSCAN
            min_samples (int): Minimum number of samples in a neighborhood for DBSCAN
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing the trained model, metrics, and preprocessing objects
        """
        # Preprocess data
        X_scaled = self._preprocess_data(X)
        
        # Initialize model based on type
        if model_type == 'kmeans':
            if n_clusters is None:
                # Try to estimate optimal number of clusters
                max_clusters = min(10, X.shape[0] // 5)  # Limit to reasonable number
                silhouette_scores = []
                
                for k in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=k, random_state=random_state)
                    labels = kmeans.fit_predict(X_scaled)
                    score = silhouette_score(X_scaled, labels)
                    silhouette_scores.append((k, score))
                
                # Choose k with highest silhouette score
                n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
            
            model = KMeans(n_clusters=n_clusters, random_state=random_state)
            labels = model.fit_predict(X_scaled)
            
            # Calculate silhouette score
            silhouette = silhouette_score(X_scaled, labels)
            metrics = {'silhouette_score': silhouette, 'n_clusters': n_clusters}
            
        elif model_type == 'dbscan':
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)
            
            # Calculate metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise points
            if n_clusters > 1:  # Silhouette score requires at least 2 clusters
                # Filter out noise points for silhouette calculation
                mask = labels != -1
                if np.sum(mask) > n_clusters:  # Need more points than clusters
                    silhouette = silhouette_score(X_scaled[mask], labels[mask])
                else:
                    silhouette = 0
            else:
                silhouette = 0
            
            metrics = {
                'silhouette_score': silhouette,
                'n_clusters': n_clusters,
                'noise_proportion': np.sum(labels == -1) / len(labels)
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Save the model
        timestamp = int(time.time())
        model_filename = f"{model_type}_clustering_{timestamp}.pkl"
        model_path = os.path.join(self.model_dir, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': self.scaler,
                'metrics': metrics,
                'labels': labels.tolist()
            }, f)
        
        return {
            'model': model,
            'metrics': metrics,
            'model_path': model_path,
            'labels': labels
        }
    
    def train_anomaly_detector(self, X, contamination='auto', random_state=42):
        """
        Train an anomaly detection model for identifying defects in AFM profiles.
        
        Args:
            X (numpy.ndarray): Feature array
            contamination (float or 'auto'): Expected proportion of outliers
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing the trained model and metrics
        """
        # Preprocess data
        X_scaled = self._preprocess_data(X)
        
        # Train isolation forest
        model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        model.fit(X_scaled)
        
        # Get anomaly scores and predictions
        scores = model.score_samples(X_scaled)
        predictions = model.predict(X_scaled)  # 1 for inliers, -1 for outliers
        
        # Convert to 0 for normal, 1 for anomaly for easier interpretation
        anomalies = np.where(predictions == -1, 1, 0)
        
        # Calculate metrics
        metrics = {
            'anomaly_proportion': np.mean(anomalies),
            'mean_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores)
        }
        
        # Save the model
        timestamp = int(time.time())
        model_filename = f"isolation_forest_{timestamp}.pkl"
        model_path = os.path.join(self.model_dir, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': self.scaler,
                'metrics': metrics
            }, f)
        
        return {
            'model': model,
            'metrics': metrics,
            'model_path': model_path,
            'anomaly_scores': -scores,  # Invert so higher = more anomalous
            'anomalies': anomalies
        }
    
    def optimize_hyperparameters(self, X, y, model_type='random_forest', param_grid=None, 
                               cv=5, scoring='f1_weighted'):
        """
        Perform hyperparameter optimization using grid search.
        
        Args:
            X (numpy.ndarray): Feature array
            y (numpy.ndarray): Label array (for supervised models)
            model_type (str): Type of model to optimize
            param_grid (dict): Grid of hyperparameters to search
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric for evaluation
            
        Returns:
            dict: Best hyperparameters and cross-validation results
        """
        # Preprocess data
        X_scaled, y_encoded = self._preprocess_data(X, y)
        
        # Define default parameter grids if not provided
        if param_grid is None:
            if model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                base_model = RandomForestClassifier(random_state=42)
            elif model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.1, 0.01],
                    'kernel': ['rbf', 'linear', 'poly']
                }
                base_model = SVC(probability=True, random_state=42)
            else:
                raise ValueError(f"Unsupported model type for optimization: {model_type}")
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,  # Use all available cores
            verbose=1
        )
        grid_search.fit(X_scaled, y_encoded)
        
        # Return best parameters and results
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def load_model(self, model_path):
        """
        Load a previously trained model.
        
        Args:
            model_path (str): Path to the saved model file
            
        Returns:
            dict: Dictionary containing the loaded model and associated objects
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Update instance variables with loaded preprocessing objects
        if 'scaler' in model_data:
            self.scaler = model_data['scaler']
        if 'label_encoder' in model_data:
            self.label_encoder = model_data['label_encoder']
        
        return model_data
