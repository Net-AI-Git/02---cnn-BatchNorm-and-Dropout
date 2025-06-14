#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Convolutional Neural Networks with Keras - Dropout & BatchNorm Version
===============================================================================

This script demonstrates advanced CNN architectures using Keras with MNIST dataset.
It includes models with different combinations of Dropout and Batch Normalization
to improve model performance and reduce overfitting.

New Features:
- 4-layer deep CNN with increasing filters (8→16→32→64)
- Multiple variations with different Dropout and BatchNorm placements
- Comparison between baseline and enhanced models
- Comprehensive analysis of regularization techniques

Uses proper Train/Validation/Test methodology:
- Train: 48,000 samples (80% of original training data)
- Validation: 12,000 samples (20% of original training data)
- Test: 10,000 samples (separate test set, used only for final evaluation)

Model Variations:
1. Baseline 4-Layer CNN (no regularization)
2. CNN with Dropout only
3. CNN with BatchNorm only
4. CNN with both Dropout and BatchNorm (after Conv)
5. CNN with both Dropout and BatchNorm (before Conv)
6. CNN with advanced regularization pattern

Installation Requirements:
--------------------------
pip install numpy==2.0.2
pip install pandas==2.2.2
pip install tensorflow-cpu==2.18.0
pip install matplotlib==3.9.2
pip install scikit-learn
"""

import os
import warnings

# Suppress TensorFlow warning messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Import required libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def create_results_directory():
    """
    Create a directory to store all result plots.

    Returns:
        str: Path to the created directory
    """
    dir_name = "Enhanced_CNN_Results"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print(f"Results will be saved to: {dir_name}")
    return dir_name


def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset with proper Train/Validation/Test split.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, num_classes)
    """
    print("Loading MNIST dataset...")

    # Load data
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    # Split training data into train and validation (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    print(f"Original training data: {X_train_full.shape[0]} samples")
    print(f"After split - Training: {X_train.shape[0]} samples")
    print(f"After split - Validation: {X_val.shape[0]} samples")
    print(f"Test data: {X_test.shape[0]} samples")

    # Reshape to be [samples][height][width][channels]
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_val = X_val.reshape(X_val.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

    # Normalize pixel values to be between 0 and 1
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    # Convert target variable into binary categories (one-hot encoding)
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)

    num_classes = y_test.shape[1]

    print(f"Final data shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"Number of classes: {num_classes}")

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes


def create_baseline_4layer_cnn(num_classes):
    """
    Create a baseline 4-layer CNN with increasing filters (no regularization).
    
    Args:
        num_classes (int): Number of output classes
        
    Returns:
        keras.Model: Compiled CNN model
    """
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(8, (5, 5), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def create_dropout_only_cnn(num_classes, dropout_rate=0.25):
    """
    Create a 4-layer CNN with Dropout only.
    
    Args:
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate
        
    Returns:
        keras.Model: Compiled CNN model
    """
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(8, (5, 5), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Higher dropout before final layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def create_batchnorm_only_cnn(num_classes):
    """
    Create a 4-layer CNN with Batch Normalization only.
    
    Args:
        num_classes (int): Number of output classes
        
    Returns:
        keras.Model: Compiled CNN model
    """
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(8, (5, 5), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def create_dropout_batchnorm_after_cnn(num_classes, dropout_rate=0.25):
    """
    Create a 4-layer CNN with both Dropout and BatchNorm (BatchNorm after Conv).
    
    Args:
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate
        
    Returns:
        keras.Model: Compiled CNN model
    """
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(8, (5, 5), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def create_dropout_batchnorm_before_cnn(num_classes, dropout_rate=0.25):
    """
    Create a 4-layer CNN with both Dropout and BatchNorm (BatchNorm before activation).
    
    Args:
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate
        
    Returns:
        keras.Model: Compiled CNN model
    """
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(8, (5, 5), padding='same'),
        BatchNormalization(),
        keras.layers.Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Conv2D(16, (3, 3), padding='same'),
        BatchNormalization(),
        keras.layers.Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        keras.layers.Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        keras.layers.Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Flatten(),
        Dense(128),
        BatchNormalization(),
        keras.layers.Activation('relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def create_advanced_regularization_cnn(num_classes):
    """
    Create a 4-layer CNN with advanced regularization pattern.
    Uses different dropout rates and strategic BatchNorm placement.
    
    Args:
        num_classes (int): Number of output classes
        
    Returns:
        keras.Model: Compiled CNN model
    """
    model = Sequential([
        Input(shape=(28, 28, 1)),
        # First block - light regularization
        Conv2D(8, (5, 5), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),  # Light dropout
        
        # Second block - moderate regularization
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        # Third block - increased regularization
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
        # Fourth block - high regularization
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        
        # Dense layers with strong regularization
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.6),  # High dropout before final layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def plot_training_history(history, model_name, save_dir):
    """
    Create and save a plot showing training history with both accuracy and loss.
    
    Args:
        history: Keras training history object
        model_name (str): Name of the model
        save_dir (str): Directory to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Dynamic Y-axis scaling for accuracy
    all_acc = history.history['accuracy'] + history.history['val_accuracy']
    min_acc = min(all_acc)
    max_acc = max(all_acc)
    margin = (max_acc - min_acc) * 0.1
    ax1.set_ylim([max(0, min_acc - margin), min(1, max_acc + margin)])

    # Plot training & validation loss
    ax2.plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Dynamic Y-axis scaling for loss
    all_loss = history.history['loss'] + history.history['val_loss']
    min_loss = min(all_loss)
    max_loss = max(all_loss)
    margin = (max_loss - min_loss) * 0.1
    ax2.set_ylim([max(0, min_loss - margin), max_loss + margin])

    plt.tight_layout()

    # Save the plot
    filename = f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: {filename}")


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test,
                             epochs=15, batch_size=150, model_name="CNN", save_dir=None):
    """
    Train and evaluate a CNN model with proper Train/Validation/Test methodology.
    
    Args:
        model: Keras model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data (used only for final evaluation)
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        model_name (str): Name for logging purposes
        save_dir (str): Directory to save plots
        
    Returns:
        tuple: (trained_model, val_accuracy, test_accuracy, history)
    """
    print(f"\n{'=' * 70}")
    print(f"Training {model_name}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}")
    print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
    print(f"{'=' * 70}")

    # Train the model using validation data for monitoring
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)

    # Plot and save training history
    if save_dir:
        plot_training_history(history, model_name, save_dir)

    # Evaluate on validation set
    val_scores = model.evaluate(X_val, y_val, verbose=0)
    val_accuracy = val_scores[1]

    # Final evaluation on test set
    test_scores = model.evaluate(X_test, y_test, verbose=0)
    test_accuracy = test_scores[1]

    # Calculate overfitting measure
    final_train_acc = history.history['accuracy'][-1]
    overfitting = final_train_acc - val_accuracy

    print(f"\nFinal Results for {model_name}:")
    print(f"Training Accuracy: {final_train_acc:.4f} ({final_train_acc * 100:.2f}%)")
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Overfitting (Train-Val): {overfitting:.4f} ({overfitting * 100:.2f}%)")

    return model, val_accuracy, test_accuracy, history, final_train_acc, overfitting


def plot_comparison_results(results, save_dir):
    """
    Create comprehensive comparison plots of all models.
    
    Args:
        results (list): List of tuples containing model results
        save_dir (str): Directory to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    model_names = [r[0] for r in results]
    val_accuracies = [r[1] * 100 for r in results]
    test_accuracies = [r[2] * 100 for r in results]
    train_accuracies = [r[4] * 100 for r in results]
    overfitting = [r[5] * 100 for r in results]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Plot 1: Test Accuracy Comparison
    bars1 = ax1.bar(range(len(model_names)), test_accuracies, color=colors, alpha=0.7)
    ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels([name.replace(' ', '\n') for name in model_names], rotation=0, fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars1, test_accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Train vs Validation Accuracy
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    ax2.bar(x_pos - width/2, train_accuracies, width, label='Training', alpha=0.7, color='skyblue')
    ax2.bar(x_pos + width/2, val_accuracies, width, label='Validation', alpha=0.7, color='lightcoral')
    ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([name.replace(' ', '\n') for name in model_names], rotation=0, fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Overfitting Analysis
    bars3 = ax3.bar(range(len(model_names)), overfitting, color=colors, alpha=0.7)
    ax3.set_title('Overfitting Analysis (Train - Validation)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Overfitting (%)', fontsize=12)
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels([name.replace(' ', '\n') for name in model_names], rotation=0, fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, ovf in zip(bars3, overfitting):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{ovf:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Training Progress Comparison (Final 5 epochs avg)
    final_val_losses = []
    for r in results:
        history = r[3]
        final_val_losses.append(np.mean(history.history['val_loss'][-5:]))  # Last 5 epochs average
    
    bars4 = ax4.bar(range(len(model_names)), final_val_losses, color=colors, alpha=0.7)
    ax4.set_title('Final Validation Loss (Last 5 Epochs Avg)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Validation Loss', fontsize=12)
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels([name.replace(' ', '\n') for name in model_names], rotation=0, fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, loss in zip(bars4, final_val_losses):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    filename = "Enhanced_CNN_Comprehensive_Comparison.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved comprehensive comparison plot: {filename}")


def main():
    """
    Main function to run the enhanced CNN training and evaluation process.
    """
    print("Enhanced Convolutional Neural Networks with Dropout & BatchNorm")
    print("=" * 80)
    print("Training 6 different CNN architectures with regularization techniques")
    print("Comparing Dropout, BatchNorm, and combined approaches")
    print("=" * 80)

    # Create results directory
    save_dir = create_results_directory()

    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = load_and_preprocess_data()

    # Define model architectures
    model_configs = [
        ("Baseline 4-Layer", create_baseline_4layer_cnn),
        ("Dropout Only", create_dropout_only_cnn),
        ("BatchNorm Only", create_batchnorm_only_cnn),
        ("Dropout + BatchNorm (After)", create_dropout_batchnorm_after_cnn),
        ("Dropout + BatchNorm (Before)", create_dropout_batchnorm_before_cnn),
        ("Advanced Regularization", create_advanced_regularization_cnn)
    ]

    epochs = 15
    batch_size = 150

    # Store all results for comparison
    all_results = []

    # Train all models
    for i, (model_name, model_creator) in enumerate(model_configs, 1):
        print(f"\n{'=' * 80}")
        print(f"MODEL {i}/{len(model_configs)}: {model_name}")
        print(f"{'=' * 80}")

        # Create model
        model = model_creator(num_classes)

        # Show model architecture
        print(f"\nModel Architecture ({model_name}):")
        model.summary()

        # Train and evaluate
        trained_model, val_accuracy, test_accuracy, history, train_accuracy, overfitting = train_and_evaluate_model(
            model, X_train, y_train, X_val, y_val, X_test, y_test,
            epochs=epochs, batch_size=batch_size,
            model_name=model_name, save_dir=save_dir
        )

        # Store results
        all_results.append((model_name, val_accuracy, test_accuracy, history, train_accuracy, overfitting))

    # Create comprehensive comparison plots
    print(f"\n{'=' * 80}")
    print("CREATING COMPREHENSIVE COMPARISON PLOTS")
    print(f"{'=' * 80}")

    plot_comparison_results(all_results, save_dir)

    # Print final summary
    print(f"\n{'=' * 90}")
    print("FINAL SUMMARY - REGULARIZATION TECHNIQUES COMPARISON")
    print(f"{'=' * 90}")

    print(f"{'Model':<35} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10} {'Overfitting':<12} {'Best For'}")
    print("-" * 95)

    best_test = max(all_results, key=lambda x: x[2])
    best_generalization = min(all_results, key=lambda x: x[5])  # Lowest overfitting
    best_val = max(all_results, key=lambda x: x[1])

    for model_name, val_acc, test_acc, history, train_acc, overfitting in all_results:
        best_indicators = []
        if model_name == best_test[0]:
            best_indicators.append("Test Acc")
        if model_name == best_generalization[0]:
            best_indicators.append("Generalization")
        if model_name == best_val[0]:
            best_indicators.append("Val Acc")
        
        best_str = ", ".join(best_indicators) if best_indicators else ""
        
        print(f"{model_name:<35} {train_acc*100:>6.2f}%  {val_acc*100:>6.2f}%  {test_acc*100:>6.2f}%  {overfitting*100:>8.2f}%    {best_str}")

    print(f"\n{'KEY INSIGHTS:'}")
    print(f"=" * 40)
    print(f"🏆 Best Test Accuracy: {best_test[0]} ({best_test[2]*100:.2f}%)")
    print(f"🎯 Best Generalization: {best_generalization[0]} (Overfitting: {best_generalization[5]*100:.2f}%)")
    print(f"📈 Best Validation: {best_val[0]} ({best_val[1]*100:.2f}%)")
    
    # Calculate improvement over baseline
    baseline_result = all_results[0]  # First model is baseline
    best_improvement = ((best_test[2] - baseline_result[2]) * 100)
    print(f"💡 Improvement over baseline: {best_improvement:.2f} percentage points")

    print(f"\nAll results saved to: {save_dir}")
    print(f"Total files created: {len(all_results) + 1} PNG files")
    print(f"- {len(all_results)} individual model training plots")
    print(f"- 1 comprehensive comparison plot")

    print(f"\n📊 METHODOLOGY:")
    print(f"- Training: {X_train.shape[0]} samples")
    print(f"- Validation: {X_val.shape[0]} samples (for monitoring)")
    print(f"- Test: {X_test.shape[0]} samples (for final evaluation)")
    print(f"- Epochs: {epochs}, Batch Size: {batch_size}")


if __name__ == "__main__":
    main()
