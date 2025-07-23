"""
Basic Text Classification Example

This example demonstrates how to use torchTextClassifiers for binary
text classification using the FastText classifier.
"""

import numpy as np
from torchTextClassifiers import create_fasttext

def main():
    print("🚀 Basic Text Classification Example")
    print("=" * 50)
    
    # Create sample data
    print("📝 Creating sample data...")
    X_train = np.array([
        "I love this product! It's amazing and works perfectly.",
        "This is terrible. Worst purchase ever made.",
        "Great quality and fast shipping. Highly recommend!",
        "Poor quality, broke after one day. Very disappointed.",
        "Excellent customer service and great value for money.",
        "Overpriced and doesn't work as advertised.",
        "Perfect! Exactly what I was looking for.",
        "Waste of money. Should have read reviews first.",
        "Outstanding product with excellent build quality.",
        "Cheap plastic, feels like it will break soon."
    ])
    
    y_train = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # 1=positive, 0=negative
    
    # Validation data
    X_val = np.array([
        "Good product, satisfied with purchase.",
        "Not worth the money, poor quality."
    ])
    y_val = np.array([1, 0])
    
    # Test data
    X_test = np.array([
        "This is an amazing product with great features!",
        "Completely disappointed with this purchase.",
        "Excellent build quality and works as expected."
    ])
    y_test = np.array([1, 0, 1])
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create FastText classifier
    print("\n🏗️ Creating FastText classifier...")
    classifier = create_fasttext(
        embedding_dim=50,
        sparse=False,
        num_tokens=5000,
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
    
    # Train the model
    print("\n🎯 Training model...")
    classifier.train(
        X_train, y_train, X_val, y_val,
        num_epochs=20,
        batch_size=4,
        patience_train=5,
        verbose=True
    )
    print("✅ Training completed!")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    predictions = classifier.predict(X_test)
    print(f"Predictions: {predictions}")
    print(f"True labels: {y_test}")
    
    # Calculate accuracy
    accuracy = classifier.validate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.3f}")
    
    # Show detailed results
    print("\n📊 Detailed Results:")
    print("-" * 40)
    for i, (text, pred, true) in enumerate(zip(X_test, predictions, y_test)):
        sentiment = "Positive" if pred == 1 else "Negative"
        correct = "✅" if pred == true else "❌"
        print(f"{i+1}. {correct} Predicted: {sentiment}")
        print(f"   Text: {text[:50]}...")
        print()
    
    print("\n🎉 Example completed successfully!")

if __name__ == "__main__":
    main()