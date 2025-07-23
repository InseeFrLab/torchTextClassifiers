"""
Advanced Training Configuration Example

This example demonstrates advanced training configurations including
custom PyTorch Lightning trainer parameters, different optimizers,
and training monitoring.
"""

import numpy as np
from torchTextClassifiers import create_fasttext

def main():
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
    
    # Create FastText classifier
    print("\n🏗️ Creating FastText classifier...")
    classifier = create_fasttext(
        embedding_dim=100,
        sparse=False,
        num_tokens=10000,
        min_count=1,
        min_n=3,
        max_n=6,
        len_word_ngrams=2,
        num_classes=2
    )
    
    # Build the model
    print("\n🔨 Building model...")
    classifier.build(X_train, y_train)
    print("✅ Model built successfully!")
    
    # Example 1: Basic training with default settings
    print("\n🎯 Example 1: Basic training with default settings...")
    classifier.train(
        X_train, y_train, X_val, y_val,
        num_epochs=15,
        batch_size=8,
        patience_train=5,
        verbose=True
    )
    
    basic_accuracy = classifier.validate(X_test, y_test)
    print(f"✅ Basic training completed! Accuracy: {basic_accuracy:.3f}")
    
    # Example 2: Advanced training with custom Lightning trainer parameters
    print("\n🚀 Example 2: Advanced training with custom parameters...")
    
    # Create a new classifier for comparison
    advanced_classifier = create_fasttext(
        embedding_dim=100,
        sparse=False,
        num_tokens=10000,
        min_count=1,
        min_n=3,
        max_n=6,
        len_word_ngrams=2,
        num_classes=2
    )
    advanced_classifier.build(X_train, y_train)
    
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
    
    advanced_classifier.train(
        X_train, y_train, X_val, y_val,
        num_epochs=20,
        batch_size=4,  # Smaller batch size with grad accumulation
        patience_train=7,
        trainer_params=advanced_trainer_params,
        verbose=True
    )
    
    advanced_accuracy = advanced_classifier.validate(X_test, y_test)
    print(f"✅ Advanced training completed! Accuracy: {advanced_accuracy:.3f}")
    
    # Example 3: Training with CPU-only (useful for small datasets or debugging)
    print("\n💻 Example 3: CPU-only training...")
    
    cpu_classifier = create_fasttext(
        embedding_dim=64,  # Smaller embedding for faster CPU training
        sparse=True,       # Sparse embeddings for efficiency
        num_tokens=5000,
        min_count=1,
        min_n=3,
        max_n=6,
        len_word_ngrams=2,
        num_classes=2
    )
    cpu_classifier.build(X_train, y_train)
    
    cpu_classifier.train(
        X_train, y_train, X_val, y_val,
        num_epochs=10,
        batch_size=16,  # Larger batch size for CPU
        cpu_run=True,   # Force CPU usage
        num_workers=0,  # No multiprocessing for CPU
        patience_train=3,
        verbose=True
    )
    
    cpu_accuracy = cpu_classifier.validate(X_test, y_test)
    print(f"✅ CPU training completed! Accuracy: {cpu_accuracy:.3f}")
    
    # Example 4: Custom training with specific Lightning callbacks
    print("\n🔧 Example 4: Training with custom callbacks...")
    
    custom_classifier = create_fasttext(
        embedding_dim=128,
        sparse=False,
        num_tokens=8000,
        min_count=1,
        min_n=3,
        max_n=6,
        len_word_ngrams=2,
        num_classes=2
    )
    custom_classifier.build(X_train, y_train)
    
    # Custom trainer with specific monitoring and checkpointing
    custom_trainer_params = {
        'max_epochs': 25,
        'enable_progress_bar': True,
        'log_every_n_steps': 1,
        'check_val_every_n_epoch': 2,  # Validate every 2 epochs
        'enable_checkpointing': True,
        'enable_model_summary': True,
    }
    
    custom_classifier.train(
        X_train, y_train, X_val, y_val,
        num_epochs=25,
        batch_size=6,
        patience_train=8,
        trainer_params=custom_trainer_params,
        verbose=True
    )
    
    custom_accuracy = custom_classifier.validate(X_test, y_test)
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