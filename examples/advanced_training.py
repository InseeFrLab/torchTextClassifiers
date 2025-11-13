"""
Advanced Training Configuration Example

This example demonstrates advanced training configurations including
custom PyTorch Lightning trainer parameters, different optimizers,
and training monitoring.
"""

import os
import random
import warnings

import numpy as np
import torch
from pytorch_lightning import seed_everything

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer

def main():
    # Set seed for reproducibility
    SEED = 42

    # Set environment variables for full reproducibility
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Use PyTorch Lightning's seed_everything for comprehensive seeding
    seed_everything(SEED, workers=True)

    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Suppress PyTorch Lightning warnings for cleaner output
    warnings.filterwarnings(
        'ignore',
        message='.*',
        category=UserWarning,
        module='pytorch_lightning'
    )

    print("⚙️ Advanced Training Configuration Example")
    print("=" * 50)

    # Create a larger dataset for demonstrating advanced training
    print("📝 Creating training dataset...")
    
    # Generate more diverse training data
    positive_samples = [
        "Excellent product with outstanding quality and performance.",
        "Amazing value for money, highly recommend to everyone!",
        "Perfect design and functionality, exceeded expectations.",
        "Great customer service and fast delivery experience.",
        "Love this product, will definitely purchase again soon.",
        "Superior quality materials and excellent craftsmanship shown.",
        "Fantastic features and user-friendly interface design provided.",
        "Outstanding durability and reliability in daily usage.",
        "Impressive performance and excellent build quality throughout.",
        "Wonderful experience from purchase to delivery service."
    ]
    
    negative_samples = [
        "Terrible product quality, completely disappointed with purchase.",
        "Poor customer service and slow delivery times experienced.",
        "Overpriced for the quality provided, not recommended.",
        "Defective product arrived, had to return immediately.",
        "Cheap materials and poor construction quality noticed.",
        "Doesn't work as advertised, waste of money.",
        "Horrible user experience and confusing interface design.",
        "Broke after few days of normal usage.",
        "Poor value for money, better alternatives available.",
        "Disappointing performance and unreliable functionality shown."
    ]
    
    # Combine and create arrays
    X_train = np.array(positive_samples + negative_samples)
    y_train = np.array([1] * len(positive_samples) + [0] * len(negative_samples))
    
    # Validation data
    X_val = np.array([
        "Good product with decent quality for the price.",
        "Not satisfied with the purchase, poor quality.",
        "Excellent service and great product quality.",
        "Disappointed with the product performance results."
    ])
    y_val = np.array([1, 0, 1, 0])
    
    # Test data
    X_test = np.array([
        "Outstanding product with amazing features!",
        "Terrible quality, complete waste of money.",
        "Great value and excellent customer support."
    ])
    y_test = np.array([1, 0, 1])
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Create and train tokenizer (shared across all examples)
    print("\n🏗️ Creating and training WordPiece tokenizer...")
    tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=128)
    training_corpus = X_train.tolist()
    tokenizer.train(training_corpus)
    print("✅ Tokenizer trained successfully!")

    # Example 1: Basic training with default settings
    print("\n🎯 Example 1: Basic training with default settings...")

    model_config = ModelConfig(
        embedding_dim=100,
        num_classes=2
    )

    classifier = torchTextClassifiers(
        tokenizer=tokenizer,
        model_config=model_config
    )
    print("✅ Classifier created successfully!")

    training_config = TrainingConfig(
        num_epochs=15,
        batch_size=8,
        lr=1e-3,
        patience_early_stopping=5,
        num_workers=0,
        trainer_params={'deterministic': True}
    )

    classifier.train(
        X_train, y_train, X_val, y_val,
        training_config=training_config,
        verbose=True
    )

    result = classifier.predict(X_test)
    basic_predictions = result["prediction"].squeeze().numpy()
    basic_accuracy = (basic_predictions == y_test).mean()
    print(f"✅ Basic training completed! Accuracy: {basic_accuracy:.3f}")
    
    # Example 2: Advanced training with custom Lightning trainer parameters
    print("\n🚀 Example 2: Advanced training with custom parameters...")

    # Create a new classifier for comparison
    advanced_model_config = ModelConfig(
        embedding_dim=100,
        num_classes=2
    )

    advanced_classifier = torchTextClassifiers(
        tokenizer=tokenizer,
        model_config=advanced_model_config
    )
    print("✅ Advanced classifier created successfully!")

    # Custom trainer parameters for advanced features
    advanced_trainer_params = {
        'accelerator': 'auto',  # Use GPU if available, else CPU
        'precision': 32,        # Use 32-bit precision
        'gradient_clip_val': 1.0,  # Gradient clipping
        'accumulate_grad_batches': 2,  # Gradient accumulation
        'deterministic': True,  # For reproducible results
        'enable_progress_bar': True,  # Show progress bar
        'log_every_n_steps': 5,  # Log every 5 steps
    }

    advanced_training_config = TrainingConfig(
        num_epochs=20,
        batch_size=4,  # Smaller batch size with grad accumulation
        lr=1e-3,
        patience_early_stopping=7,
        num_workers=0,
        cpu_run=False,  # Don't override accelerator from trainer_params
        trainer_params=advanced_trainer_params
    )

    advanced_classifier.train(
        X_train, y_train, X_val, y_val,
        training_config=advanced_training_config,
        verbose=True
    )

    advanced_result = advanced_classifier.predict(X_test)
    advanced_predictions = advanced_result["prediction"].squeeze().numpy()
    advanced_accuracy = (advanced_predictions == y_test).mean()
    print(f"✅ Advanced training completed! Accuracy: {advanced_accuracy:.3f}")
    
    # Example 3: Training with CPU-only (useful for small datasets or debugging)
    print("\n💻 Example 3: CPU-only training...")

    cpu_model_config = ModelConfig(
        embedding_dim=64,  # Smaller embedding for faster CPU training
        num_classes=2
    )

    cpu_classifier = torchTextClassifiers(
        tokenizer=tokenizer,
        model_config=cpu_model_config
    )
    print("✅ CPU classifier created successfully!")

    cpu_training_config = TrainingConfig(
        num_epochs=10,
        batch_size=16,  # Larger batch size for CPU
        lr=1e-3,
        patience_early_stopping=3,
        cpu_run=False,  # Don't override accelerator from trainer_params
        num_workers=0,  # No multiprocessing for CPU
        trainer_params={'deterministic': True, 'accelerator': 'cpu'}
    )

    cpu_classifier.train(
        X_train, y_train, X_val, y_val,
        training_config=cpu_training_config,
        verbose=True
    )

    cpu_result = cpu_classifier.predict(X_test)
    cpu_predictions = cpu_result["prediction"].squeeze().numpy()
    cpu_accuracy = (cpu_predictions == y_test).mean()
    print(f"✅ CPU training completed! Accuracy: {cpu_accuracy:.3f}")
    
    # Example 4: Custom training with specific Lightning callbacks
    print("\n🔧 Example 4: Training with custom callbacks...")

    custom_model_config = ModelConfig(
        embedding_dim=128,
        num_classes=2
    )

    custom_classifier = torchTextClassifiers(
        tokenizer=tokenizer,
        model_config=custom_model_config
    )
    print("✅ Custom classifier created successfully!")

    # Custom trainer with specific monitoring and checkpointing
    custom_trainer_params = {
        'max_epochs': 25,
        'enable_progress_bar': True,
        'log_every_n_steps': 1,
        'check_val_every_n_epoch': 2,  # Validate every 2 epochs
        'enable_checkpointing': True,
        'enable_model_summary': True,
        'deterministic': True,
    }

    custom_training_config = TrainingConfig(
        num_epochs=25,
        batch_size=6,
        lr=1e-3,
        patience_early_stopping=8,
        num_workers=0,
        trainer_params=custom_trainer_params
    )

    custom_classifier.train(
        X_train, y_train, X_val, y_val,
        training_config=custom_training_config,
        verbose=True
    )

    custom_result = custom_classifier.predict(X_test)
    custom_predictions = custom_result["prediction"].squeeze().numpy()
    custom_accuracy = (custom_predictions == y_test).mean()
    print(f"✅ Custom training completed! Accuracy: {custom_accuracy:.3f}")
    
    # Compare all training approaches
    print("\n📊 Training Comparison Results:")
    print("-" * 50)
    print(f"Basic training:     {basic_accuracy:.3f}")
    print(f"Advanced training:  {advanced_accuracy:.3f}")
    print(f"CPU-only training:  {cpu_accuracy:.3f}")
    print(f"Custom training:    {custom_accuracy:.3f}")
    
    # Find best performing model
    results = {
        'Basic': basic_accuracy,
        'Advanced': advanced_accuracy,
        'CPU-only': cpu_accuracy,
        'Custom': custom_accuracy
    }
    best_method = max(results, key=results.get)
    print(f"\n🏆 Best performing method: {best_method} (Accuracy: {results[best_method]:.3f})")
    
    # Demonstrate prediction with best model
    print(f"\n🔮 Making predictions with {best_method.lower()} model...")
    best_classifier = {
        'Basic': classifier,
        'Advanced': advanced_classifier,
        'CPU-only': cpu_classifier,
        'Custom': custom_classifier
    }[best_method]
    
    predictions = best_classifier.predict(X_test)
    print("Test predictions:")
    for i, (text, pred, true) in enumerate(zip(X_test, predictions, y_test)):
        sentiment = "Positive" if pred == 1 else "Negative"
        correct = "✅" if pred == true else "❌"
        print(f"{i+1}. {correct} {sentiment}: {text[:50]}...")

    print("\n🎉 Advanced training example completed successfully!")


if __name__ == "__main__":
    main()