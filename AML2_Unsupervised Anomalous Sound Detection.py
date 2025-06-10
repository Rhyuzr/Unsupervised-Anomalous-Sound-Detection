"""
DCASE Challenge: Anomalous Sound Detection
This script implements a solution for detecting anomalous sounds from industrial machines (slide rail)
using only normal sound samples for training.

Approach: Autoencoder-based anomaly detection
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, Conv2D, MaxPooling2D, Flatten, Reshape
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DcaseAnomalyDetection:
    def __init__(self, dev_data_path, eval_data_path=None, machine_type='slider', sr=16000, duration=10,
                 n_mels=128, n_fft=1024, hop_length=512):
        """
        Initialize the anomaly detection system
        
        Args:
            dev_data_path: Path to the development dataset
            eval_data_path: Path to the evaluation dataset (optional)
            machine_type: Type of machine to analyze ('slider' in this case)
            sr: Sampling rate of audio files
            duration: Duration of audio clips in seconds
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.dev_data_path = dev_data_path
        self.eval_data_path = eval_data_path
        self.machine_type = machine_type
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, split_ratio=0.2, test_size=0.5):
        """
        Load audio files and prepare datasets
        
        Args:
            split_ratio: Ratio for splitting training data into train/validation
            test_size: Portion of test data to use (since we need to create our own test set)
            
        Returns:
            X_train, X_val, X_test, y_test: Prepared datasets
        """
        print("Loading and processing data...")
        
        # Paths for dataset
        dev_train_path = os.path.join(self.dev_data_path, 'dev_data', self.machine_type, 'train')
        dev_test_path = os.path.join(self.dev_data_path, 'dev_data', self.machine_type, 'test')
        
        # Lists to store data
        normal_train_features = []
        normal_test_features = []
        anomaly_test_features = []
        
        # Process training data (all normal samples)
        normal_train_files = glob.glob(os.path.join(dev_train_path, 'normal_*.wav'))
        print(f"Found {len(normal_train_files)} normal training files")
        
        for file_path in tqdm(normal_train_files, desc="Processing normal training files"):
            features = self.extract_features(file_path)
            normal_train_features.append(features)
        
        # Process test data (both normal and anomalous)
        normal_test_files = glob.glob(os.path.join(dev_test_path, 'normal_*.wav'))
        anomaly_test_files = glob.glob(os.path.join(dev_test_path, 'anomaly_*.wav'))
        
        print(f"Found {len(normal_test_files)} normal test files and {len(anomaly_test_files)} anomaly test files")
        
        for file_path in tqdm(normal_test_files, desc="Processing normal test files"):
            features = self.extract_features(file_path)
            normal_test_features.append(features)
        
        for file_path in tqdm(anomaly_test_files, desc="Processing anomaly test files"):
            features = self.extract_features(file_path)
            anomaly_test_features.append(features)
        
        # Convert to numpy arrays
        normal_train_features = np.array(normal_train_features)
        normal_test_features = np.array(normal_test_features)
        anomaly_test_features = np.array(anomaly_test_features)
        
        # Since we don't have official test labels, we'll create our own validation and test sets
        # Split normal training data into train and validation
        X_train, X_val = train_test_split(normal_train_features, test_size=split_ratio, random_state=42)
        
        # Combine normal and anomaly test samples and create labels
        # Select a subset of test data to create a balanced test set
        n_normal_test = int(len(normal_test_features) * test_size)
        n_anomaly_test = int(len(anomaly_test_features) * test_size)
        
        # Select samples
        selected_normal = normal_test_features[:n_normal_test]
        selected_anomaly = anomaly_test_features[:n_anomaly_test]
        
        # Combine and create labels
        X_test = np.concatenate([selected_normal, selected_anomaly])
        y_test = np.concatenate([np.zeros(n_normal_test), np.ones(n_anomaly_test)])
        
        # Standardize features based on training data
        self.scaler.fit(X_train.reshape(X_train.shape[0], -1))
        
        X_train = self.scaler.transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
        X_val = self.scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
        X_test = self.scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
        
        print(f"Data loaded: {X_train.shape[0]} train, {X_val.shape[0]} validation, {X_test.shape[0]} test samples")
        print(f"Test set has {np.sum(y_test == 0)} normal and {np.sum(y_test == 1)} anomaly samples")
        
        return X_train, X_val, X_test, y_test
    
    def extract_features(self, file_path):
        """
        Extract mel spectrogram features from audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            mel_spectrogram: Extracted features
        """
        # Load audio file
        y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)
        
        # Pad if audio is shorter than expected duration
        if len(y) < self.sr * self.duration:
            y = np.pad(y, (0, self.sr * self.duration - len(y)), 'constant')
        
        # Extract mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        
        # Convert to decibels
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        return mel_spectrogram
    
    def build_autoencoder(self):
        """
        Build a reliable autoencoder model for anomaly detection
        """
        # Get input shape from a sample
        sample_file = glob.glob(os.path.join(self.dev_data_path, 'dev_data', self.machine_type, 'train', 'normal_*.wav'))[0]
        sample_features = self.extract_features(sample_file)
        input_shape = sample_features.shape
        
        print(f"Input shape: {input_shape}")
        
        # Build a simple but effective autoencoder model
        input_img = Input(shape=(input_shape[0], input_shape[1]))
        
        # Reshape to have a single channel for conv layers
        x = Reshape((input_shape[0], input_shape[1], 1))(input_img)
        
        # Encoder - simple and reliable architecture
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        
        # Decoder - matching architecture for proper reconstruction
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)
        
        # Final convolution with proper padding to ensure output shape matches input
        x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        # Handle potential shape mismatches
        # First get the current shape
        current_shape = x.shape[1:3]
        target_shape = input_shape
        
        # Create a Lambda layer to handle dynamic resizing
        def resize_to_target(tensor):
            # Get the batch size dynamically
            batch_size = tf.shape(tensor)[0]
            # Reshape to remove the channel dimension
            reshaped = tf.reshape(tensor, [batch_size, current_shape[0], current_shape[1]])
            # Use resize to get exact dimensions needed
            if current_shape != target_shape:
                reshaped = tf.image.resize(tf.expand_dims(reshaped, -1), [target_shape[0], target_shape[1]])
                reshaped = tf.squeeze(reshaped, -1)
            return reshaped
        
        decoded = tf.keras.layers.Lambda(resize_to_target)(x)
        
        # Create model
        autoencoder = Model(input_img, decoded)
        
        # Use standard MSE loss which works well for this problem
        autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
        
        self.model = autoencoder
        print(autoencoder.summary())
        
        return autoencoder
    
    def train_model(self, X_train, X_val, epochs=100, batch_size=32):
        """
        Train the autoencoder model
        
        Args:
            X_train: Training data
            X_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            history: Training history
        """
        print("Training model...")
        
        # Build model if not already built
        if self.model is None:
            self.build_autoencoder()
        
        # Simple callbacks for reliable training
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=5,  
                restore_best_weights=True,
                verbose=1
            ),
            # Model checkpoint to save best model
            ModelCheckpoint(
                'best_autoencoder.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model with straightforward approach
        history = self.model.fit(
            X_train, X_train,  # Input and target are the same for autoencoder
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def compute_anomaly_scores(self, X):
        """
        Compute anomaly scores as reconstruction error
        
        Args:
            X: Input data
            
        Returns:
            scores: Anomaly scores
        """
        # Get reconstructions
        reconstructions = self.model.predict(X)
        
        # Compute MSE for each sample
        mse = np.mean(np.square(X - reconstructions), axis=(1, 2))
        
        return mse
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            auc: Area under ROC curve
        """
        print("Evaluating model...")
        
        # Compute anomaly scores
        anomaly_scores = self.compute_anomaly_scores(X_test)
        
        # Compute AUC
        auc = roc_auc_score(y_test, anomaly_scores)
        print(f"AUC: {auc:.4f}")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, 'b', label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig('roc_curve.png')
        plt.close()
        
        # Plot anomaly score distributions
        plt.figure(figsize=(10, 8))
        sns.histplot(anomaly_scores[y_test == 0], kde=True, label='Normal', alpha=0.5)
        sns.histplot(anomaly_scores[y_test == 1], kde=True, label='Anomaly', alpha=0.5)
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        plt.savefig('anomaly_score_distribution.png')
        plt.close()
        
        return auc
    
    def predict(self, file_path, threshold=None):
        """
        Predict if a sound sample is anomalous
        
        Args:
            file_path: Path to audio file
            threshold: Threshold for anomaly detection (optional)
            
        Returns:
            is_anomaly: Boolean indicating if the sample is anomalous
            anomaly_score: Computed anomaly score
        """
        # Extract features
        features = self.extract_features(file_path)
        features = features.reshape(1, *features.shape)
        
        # Standardize features
        features = self.scaler.transform(features.reshape(features.shape[0], -1)).reshape(features.shape)
        
        # Compute anomaly score
        anomaly_score = self.compute_anomaly_scores(features)[0]
        
        # Determine if anomalous
        if threshold is not None:
            is_anomaly = anomaly_score > threshold
            return is_anomaly, anomaly_score
        else:
            return anomaly_score
    
    def visualize_reconstructions(self, X_test, y_test, n_samples=5):
        """
        Visualize original vs reconstructed spectrograms
        
        Args:
            X_test: Test data
            y_test: Test labels
            n_samples: Number of samples to visualize
        """
        # Get reconstructions
        reconstructions = self.model.predict(X_test)
        
        # Compute anomaly scores
        anomaly_scores = self.compute_anomaly_scores(X_test)
        
        # Select random samples from normal and anomalous classes
        normal_indices = np.where(y_test == 0)[0]
        anomaly_indices = np.where(y_test == 1)[0]
        
        normal_samples = np.random.choice(normal_indices, min(n_samples, len(normal_indices)), replace=False)
        anomaly_samples = np.random.choice(anomaly_indices, min(n_samples, len(anomaly_indices)), replace=False)
        
        # Plot normal samples
        plt.figure(figsize=(15, n_samples * 5))
        for i, idx in enumerate(normal_samples):
            # Original
            plt.subplot(n_samples, 2, i*2 + 1)
            librosa.display.specshow(X_test[idx], sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='mel')
            plt.title(f'Original (Normal) - Score: {anomaly_scores[idx]:.4f}')
            plt.colorbar(format='%+2.0f dB')
            
            # Reconstruction
            plt.subplot(n_samples, 2, i*2 + 2)
            librosa.display.specshow(reconstructions[idx], sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='mel')
            plt.title(f'Reconstructed (Normal)')
            plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig('normal_reconstructions.png')
        plt.close()
        
        # Plot anomalous samples
        plt.figure(figsize=(15, n_samples * 5))
        for i, idx in enumerate(anomaly_samples):
            # Original
            plt.subplot(n_samples, 2, i*2 + 1)
            librosa.display.specshow(X_test[idx], sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='mel')
            plt.title(f'Original (Anomaly) - Score: {anomaly_scores[idx]:.4f}')
            plt.colorbar(format='%+2.0f dB')
            
            # Reconstruction
            plt.subplot(n_samples, 2, i*2 + 2)
            librosa.display.specshow(reconstructions[idx], sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='mel')
            plt.title(f'Reconstructed (Anomaly)')
            plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig('anomaly_reconstructions.png')
        plt.close()


def main():
    """
    Main function to run the anomaly detection system
    """
    # Set data paths based on the user's structure
    dev_data_path = 's:/doc/aml/2/dev_data'
    eval_data_path = 's:/doc/aml/2/eval_data'
    
    # Check if dataset exists
    if not os.path.exists(os.path.join(dev_data_path, 'dev_data', 'slider')):
        print(f"\nError: Dataset not found at {dev_data_path}/dev_data/slider")
        print("\nPlease ensure the dataset has the correct structure.")
        return
    
    try:
        # Initialize system
        system = DcaseAnomalyDetection(dev_data_path=dev_data_path, eval_data_path=eval_data_path, machine_type='slider')
        
        # Load and prepare data
        X_train, X_val, X_test, y_test = system.load_data(split_ratio=0.2)
        
        # Build and train model
        system.build_autoencoder()
        history = system.train_model(X_train, X_val, epochs=15, batch_size=32)
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('training_history.png')
        plt.close()
        
        # Evaluate model
        auc = system.evaluate(X_test, y_test)
        
        # Visualize reconstructions
        system.visualize_reconstructions(X_test, y_test, n_samples=3)
        
        print(f"Final AUC: {auc:.4f}")
        print("Done! Check the generated plots for visual analysis.")
        
        # Optionally, if you want to evaluate on the evaluation dataset
        if eval_data_path and os.path.exists(os.path.join(eval_data_path, 'eval_data', 'slider')):
            print("\nEvaluating on evaluation dataset...")
            # This would require additional code to process the evaluation dataset
            # Since the evaluation dataset doesn't have labels, we'll need to adapt our approach
            # This is left as an extension for later
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        print("Please check if the dataset structure is correct and try again.")
        print("If the issue persists, you may need to adjust the code parameters to match your dataset.")


if __name__ == "__main__":
    main()
