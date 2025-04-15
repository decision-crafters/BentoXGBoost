import os
import re
import glob
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import logging
import xgboost as xgb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    A class to process data for XGBoost training.
    """
    
    def __init__(self, max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize the DataProcessor.
        
        Args:
            max_features: Maximum number of features to extract from text
            ngram_range: Range of n-grams to consider
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.scaler = StandardScaler(with_mean=False)  # Sparse matrices don't support centering
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing special characters, extra whitespace, etc.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def process_text_files(self, directory: str, file_pattern: str = "**/*.md") -> List[str]:
        """
        Process all text files in a directory.
        
        Args:
            directory: Directory containing text files
            file_pattern: Pattern to match files
            
        Returns:
            List of preprocessed text from each file
        """
        file_paths = glob.glob(os.path.join(directory, file_pattern), recursive=True)
        logger.info(f"Found {len(file_paths)} files matching pattern {file_pattern} in {directory}")
        
        processed_texts = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                processed_text = self.preprocess_text(text)
                processed_texts.append(processed_text)
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
        
        logger.info(f"Processed {len(processed_texts)} text files")
        
        return processed_texts
    
    def process_markdown_list(self, markdown_texts: List[str]) -> List[str]:
        """
        Process a list of markdown texts.
        
        Args:
            markdown_texts: List of markdown texts
            
        Returns:
            List of preprocessed texts
        """
        processed_texts = [self.preprocess_text(text) for text in markdown_texts]
        logger.info(f"Processed {len(processed_texts)} markdown texts")
        
        return processed_texts
    
    def vectorize_texts(self, texts: List[str], fit: bool = True) -> np.ndarray:
        """
        Convert texts to feature vectors.
        
        Args:
            texts: List of texts to vectorize
            fit: Whether to fit the vectorizer on the texts
            
        Returns:
            Feature matrix
        """
        if fit:
            logger.info(f"Fitting vectorizer on {len(texts)} texts")
            X = self.vectorizer.fit_transform(texts)
        else:
            logger.info(f"Transforming {len(texts)} texts with pre-fitted vectorizer")
            X = self.vectorizer.transform(texts)
        
        logger.info(f"Vectorized texts to shape {X.shape}")
        
        return X
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale features to have zero mean and unit variance.
        
        Args:
            X: Feature matrix
            fit: Whether to fit the scaler on the features
            
        Returns:
            Scaled feature matrix
        """
        if fit:
            logger.info(f"Fitting scaler on features of shape {X.shape}")
            X_scaled = self.scaler.fit_transform(X)
        else:
            logger.info(f"Transforming features of shape {X.shape} with pre-fitted scaler")
            X_scaled = self.scaler.transform(X)
        
        logger.info(f"Scaled features to shape {X_scaled.shape}")
        
        return X_scaled
    
    def create_binary_labels(self, num_samples: int, positive_ratio: float = 0.5) -> np.ndarray:
        """
        Create binary labels for unsupervised data.
        
        Args:
            num_samples: Number of samples
            positive_ratio: Ratio of positive samples
            
        Returns:
            Binary labels
        """
        num_positive = int(num_samples * positive_ratio)
        labels = np.zeros(num_samples)
        labels[:num_positive] = 1
        np.random.shuffle(labels)
        
        logger.info(f"Created binary labels with {num_positive} positive samples out of {num_samples}")
        
        return labels
    
    def prepare_data_for_xgboost(self, X: np.ndarray, y: np.ndarray) -> xgb.DMatrix:
        """
        Prepare data for XGBoost training.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            XGBoost DMatrix
        """
        logger.info(f"Preparing XGBoost DMatrix with {X.shape[0]} samples and {X.shape[1]} features")
        return xgb.DMatrix(X, label=y)
    
    def process_directory_for_training(self, directory: str, file_pattern: str = "**/*.md", 
                                      positive_ratio: float = 0.5) -> xgb.DMatrix:
        """
        Process all text files in a directory and prepare for XGBoost training.
        
        Args:
            directory: Directory containing text files
            file_pattern: Pattern to match files
            positive_ratio: Ratio of positive samples for binary labels
            
        Returns:
            XGBoost DMatrix
        """
        texts = self.process_text_files(directory, file_pattern)
        X = self.vectorize_texts(texts)
        X_scaled = self.scale_features(X)
        y = self.create_binary_labels(X_scaled.shape[0], positive_ratio)
        
        return self.prepare_data_for_xgboost(X_scaled, y)
    
    def process_markdown_list_for_training(self, markdown_texts: List[str], 
                                          positive_ratio: float = 0.5) -> xgb.DMatrix:
        """
        Process a list of markdown texts and prepare for XGBoost training.
        
        Args:
            markdown_texts: List of markdown texts
            positive_ratio: Ratio of positive samples for binary labels
            
        Returns:
            XGBoost DMatrix
        """
        processed_texts = self.process_markdown_list(markdown_texts)
        X = self.vectorize_texts(processed_texts)
        X_scaled = self.scale_features(X)
        y = self.create_binary_labels(X_scaled.shape[0], positive_ratio)
        
        return self.prepare_data_for_xgboost(X_scaled, y)
