#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional CNN Research: BatchNorm & Dropout Placement Study on Fashion-MNIST
===============================================================================

This study investigates the effects of different regularization techniques
and their placement order in deep CNN architectures using Fashion-MNIST dataset.
We examine 6 carefully designed architectures to test specific hypotheses about 
BatchNormalization and Dropout positioning.

Research Questions:
1. How does BatchNorm placement (pre vs post-activation) affect convergence?
2. What is the optimal order of BatchNorm and Dropout operations?
3. How do different regularization strategies impact generalization?

Dataset: Fashion-MNIST (28x28 grayscale images of clothing items)
- 10 classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- More challenging than standard MNIST digits
- Same input dimensions for direct architecture comparison

Models Under Investigation:
1. Modern Baseline - No regularization control
2. Post-Activation BatchNorm - Classical BN placement (Ioffe & Szegedy, 2015)
3. Pre-Activation BatchNorm - Modern BN placement (He et al., 2016)
4. BatchNorm-First - BN before Dropout hypothesis
5. Dropout-First - Dropout before BN hypothesis
6. Classical Regularization - Conv → ReLU → BatchNorm → Dropout sequence

Requirements:
pip install tensorflow numpy matplotlib scikit-learn
"""

import os
import warnings
from typing import Tuple, List, Dict, Any

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


class CNNResearchPipeline:
    """
    Professional CNN research pipeline for BatchNorm and Dropout placement study.
    
    This class encapsulates the entire research workflow including data preparation,
    model creation, training, evaluation, and visualization.
    """
    
    def __init__(self, results_dir: str = "Fashion_MNIST_Research_Results"):
        """
        Initialize the research pipeline.
        
        Args:
            results_dir: Directory name for saving results and plots
        """
        self.results_dir = results_dir
        self.setup_results_directory()
        
        # Research configuration
        self.epochs = 12
        self.batch_size = 128
        self.validation_split = 0.2
        self.random_seed = 42
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        
        print("=" * 80)
        print("Professional CNN Research: BatchNorm & Dropout Study on Fashion-MNIST")
        print("=" * 80)
        
    def setup_results_directory(self) -> None:
        """Create results directory if it doesn't exist."""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        print(f"Results will be saved to: {self.results_dir}")
        
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, ...]:
        """
        Load and preprocess Fashion-MNIST dataset with proper train/validation/test split.
        
        Fashion-MNIST contains 28x28 grayscale images of 10 clothing categories:
        0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat,
        5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, num_classes)
        """
        print("\n📊 Loading and preprocessing Fashion-MNIST dataset...")
        
        # Load Fashion-MNIST data
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
        
        # Create stratified train/validation split to ensure balanced class distribution
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=self.validation_split, 
            random_state=self.random_seed, 
            stratify=y_train_full
        )
        
        # Reshape and normalize
        X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        X_val = X_val.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        
        # One-hot encode labels
        num_classes = 10
        y_train = to_categorical(y_train, num_classes)
        y_val = to_categorical(y_val, num_classes)
        y_test = to_categorical(y_test, num_classes)
        
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Validation samples: {X_val.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        print(f"   Number of classes: {num_classes}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, num_classes
    
    def create_modern_baseline(self, num_classes: int) -> keras.Model:
        """
        Create modern baseline CNN without regularization.
        
        Architecture: Progressive filter increase (32→64→128→256) with GlobalAvgPool
        to reduce parameters while maintaining representational power.
        
        Args:
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ], name="Modern_Baseline")
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_post_activation_batchnorm(self, num_classes: int) -> keras.Model:
        """
        Create CNN with post-activation BatchNorm (Ioffe & Szegedy, 2015).
        
        Pattern: Conv → ReLU → BatchNorm
        This follows the original BatchNorm paper approach.
        
        Args:
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(num_classes, activation='softmax')
        ], name="Post_Activation_BatchNorm")
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_pre_activation_batchnorm(self, num_classes: int) -> keras.Model:
        """
        Create CNN with pre-activation BatchNorm (He et al., 2016).
        
        Pattern: Conv → BatchNorm → ReLU
        This approach claims better gradient flow in deep networks.
        
        Args:
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (5, 5), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dense(num_classes, activation='softmax')
        ], name="Pre_Activation_BatchNorm")
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_batchnorm_first_regularization(self, num_classes: int) -> keras.Model:
        """
        Create CNN with BatchNorm before Dropout.
        
        Hypothesis: Normalize first, then apply stochastic regularization.
        Pattern: Conv → BatchNorm → ReLU → Dropout
        
        Args:
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (5, 5), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.1),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.15),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.25),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation='softmax')
        ], name="BatchNorm_First_Regularization")
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_classical_regularization_cnn(self, num_classes: int) -> keras.Model:
        """
        Create CNN with classical regularization order.
        
        Pattern: Conv → ReLU → BatchNorm → Dropout
        This is a widely adopted sequence in modern CNN architectures.
        
        Args:
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.15),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation='softmax')
        ], name="Classical_Regularization")
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        """
        Create CNN with Dropout before BatchNorm.
        
        Hypothesis: Apply stochastic regularization first, then normalize.
        Pattern: Conv → ReLU → Dropout → BatchNorm
        
        Args:
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
            layers.Dropout(0.1),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Dropout(0.15),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.Dropout(0.25),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.BatchNormalization(),
            layers.Dense(num_classes, activation='softmax')
        ], name="Dropout_First_Regularization")
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> keras.callbacks.History:
        """
        Train a model with consistent parameters and early stopping.
        
        Args:
            model: Compiled Keras model
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Training history object
        """
        print(f"\n🚀 Training {model.name}...")
        print(f"   Architecture: {self._count_parameters(model):,} parameters")
        
        # Configure early stopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping],
            verbose=1,
            shuffle=True
        )
        
        return history
    
    def evaluate_model(self, model: keras.Model, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained Keras model
            X_test, y_test: Test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
        
        print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"   Test Loss: {test_loss:.4f}")
        
        return results
    
    def plot_training_results(self, history: keras.callbacks.History, 
                            model_name: str, model_index: int) -> None:
        """
        Create and save training plots with accuracy and loss side by side.
        
        Args:
            history: Training history object
            model_name: Name of the model for plot title
            model_index: Index for filename ordering
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        epochs_range = range(1, len(history.history['accuracy']) + 1)
        train_acc = [acc * 100 for acc in history.history['accuracy']]
        val_acc = [acc * 100 for acc in history.history['val_accuracy']]
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        # Plot accuracy
        ax1.plot(epochs_range, train_acc, 'b-', linewidth=2, label='Training Accuracy')
        ax1.plot(epochs_range, val_acc, 'r-', linewidth=2, label='Validation Accuracy')
        ax1.set_title(f'{model_name}\nAccuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Smart scaling for accuracy
        all_acc = train_acc + val_acc
        acc_min, acc_max = min(all_acc), max(all_acc)
        acc_margin = (acc_max - acc_min) * 0.1
        ax1.set_ylim([max(0, acc_min - acc_margin), min(100, acc_max + acc_margin)])
        
        # Plot loss
        ax2.plot(epochs_range, train_loss, 'b-', linewidth=2, label='Training Loss')
        ax2.plot(epochs_range, val_loss, 'r-', linewidth=2, label='Validation Loss')
        ax2.set_title(f'{model_name}\nLoss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epochs', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Smart scaling for loss
        all_loss = train_loss + val_loss
        loss_min, loss_max = min(all_loss), max(all_loss)
        loss_margin = (loss_max - loss_min) * 0.1
        ax2.set_ylim([max(0, loss_min - loss_margin), loss_max + loss_margin])
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{model_index}_{model_name.replace(' ', '_')}_Training.png"
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved training plot: {filename}")
    
    def plot_final_comparison(self, results: List[Dict[str, Any]]) -> None:
        """
        Create final comparison plot of test accuracies.
        
        Args:
            results: List of dictionaries containing model results
        """
        model_names = [r['name'] for r in results]
        test_accuracies = [r['test_accuracy'] * 100 for r in results]
        
        plt.figure(figsize=(12, 8))
        
        # Create bar plot with custom colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        bars = plt.bar(range(len(model_names)), test_accuracies, color=colors, alpha=0.8)
        
        # Customize plot
        plt.title('Test Accuracy Comparison on Fashion-MNIST\nBatchNorm & Dropout Placement Study', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Model Architecture', fontsize=14)
        plt.ylabel('Test Accuracy (%)', fontsize=14)
        
        # Set x-axis labels
        plt.xticks(range(len(model_names)), 
                  [name.replace(' ', '\n') for name in model_names], 
                  fontsize=11)
        
        # Smart scaling for y-axis
        acc_min, acc_max = min(test_accuracies), max(test_accuracies)
        acc_margin = (acc_max - acc_min) * 0.1
        plt.ylim([max(0, acc_min - acc_margin), min(100, acc_max + acc_margin)])
        
        # Add value labels on bars
        for bar, acc in zip(bars, test_accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{acc:.2f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save plot
        filename = "Final_Test_Accuracy_Comparison.png"
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Saved final comparison plot: {filename}")
    
    def _count_parameters(self, model: keras.Model) -> int:
        """Count trainable parameters in model."""
        return model.count_params()
    
    def run_research_study(self) -> None:
        """
        Run the complete research study.
        
        This method orchestrates the entire research pipeline:
        1. Data preparation
        2. Model creation and training
        3. Evaluation and analysis
        4. Visualization and reporting
        """
        # Prepare data
        X_train, y_train, X_val, y_val, X_test, y_test, num_classes = self.load_and_preprocess_data()
        
        # Define model configurations
        model_configs = [
            ("Modern Baseline", self.create_modern_baseline),
            ("Post-Activation BatchNorm", self.create_post_activation_batchnorm),
            ("Pre-Activation BatchNorm", self.create_pre_activation_batchnorm),
            ("BatchNorm-First Regularization", self.create_batchnorm_first_regularization),
            ("Dropout-First Regularization", self.create_dropout_first_regularization),
            ("Classical Regularization", self.create_classical_regularization_cnn)
        ]
        
        # Store results for final analysis
        all_results = []
        
        print(f"\n🔬 Starting research study with {len(model_configs)} architectures...")
        print(f"   Using early stopping (patience=3) and stratified validation split")
        
        # Train and evaluate each model
        for i, (model_name, model_creator) in enumerate(model_configs, 1):
            print(f"\n{'='*80}")
            print(f"MODEL {i}/{len(model_configs)}: {model_name}")
            print(f"{'='*80}")
            
            # Create and train model
            model = model_creator(num_classes)
            history = self.train_model(model, X_train, y_train, X_val, y_val)
            
            # Evaluate model
            test_results = self.evaluate_model(model, X_test, y_test)
            
            # Create training plots
            self.plot_training_results(history, model_name, i)
            
            # Store results
            all_results.append({
                'name': model_name,
                'history': history,
                'test_accuracy': test_results['test_accuracy'],
                'test_loss': test_results['test_loss'],
                'parameters': self._count_parameters(model)
            })
        
        # Create final comparison
        self.plot_final_comparison(all_results)
        
        # Print final summary
        self._print_research_summary(all_results)
    
    def _print_research_summary(self, results: List[Dict[str, Any]]) -> None:
        """Print comprehensive research summary."""
        print(f"\n{'='*90}")
        print("RESEARCH STUDY SUMMARY: BatchNorm & Dropout on Fashion-MNIST")
        print(f"{'='*90}")
        
        print(f"{'Model':<40} {'Parameters':<12} {'Test Acc':<10} {'Best For'}")
        print("-" * 90)
        
        # Find best models for different criteria
        best_accuracy = max(results, key=lambda x: x['test_accuracy'])
        best_efficiency = min(results, key=lambda x: x['parameters'])
        
        for result in results:
            name = result['name']
            params = f"{result['parameters']:,}"
            test_acc = f"{result['test_accuracy']*100:.2f}%"
            
            best_indicators = []
            if result == best_accuracy:
                best_indicators.append("Accuracy")
            if result == best_efficiency:
                best_indicators.append("Efficiency")
                
            best_str = ", ".join(best_indicators)
            
            print(f"{name:<40} {params:<12} {test_acc:<10} {best_str}")
        
        print(f"\n🏆 KEY FINDINGS:")
        print(f"   • Best performing model: {best_accuracy['name']} ({best_accuracy['test_accuracy']*100:.2f}%)")
        print(f"   • Most efficient model: {best_efficiency['name']} ({best_efficiency['parameters']:,} params)")
        
        # Calculate baseline improvement
        baseline = results[0]  # First model is baseline
        improvements = [(r['test_accuracy'] - baseline['test_accuracy']) * 100 
                       for r in results[1:]]
        best_improvement = max(improvements)
        
        print(f"   • Best improvement over baseline: +{best_improvement:.2f} percentage points")
        print(f"\n📁 All results saved to: {self.results_dir}")
        print(f"   • {len(results)} individual training plots")
        print(f"   • 1 comprehensive comparison plot")


def main():
    """Main function to run the CNN research study."""
    # Initialize research pipeline
    research = CNNResearchPipeline()
    
    # Run complete study
    research.run_research_study()
    
    print(f"\n✅ Research study completed successfully!")
    print(f"Check the '{research.results_dir}' directory for all results and plots.")


if __name__ == "__main__":
    main()
