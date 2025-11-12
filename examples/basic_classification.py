"""
Basic Text Classification Example

This example demonstrates how to use torchTextClassifiers for binary
text classification using the FastText classifier.
"""

import numpy as np
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer


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
    
    # Create and train tokenizer
    print("\n🏗️ Creating and training WordPiece tokenizer...")
    tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=128)
    
    # Train tokenizer on the training corpus
    training_corpus = X_train.tolist()
    tokenizer.train(training_corpus)
    print("✅ Tokenizer trained successfully!")

    # Create model configuration
    print("\n🔧 Creating model configuration...")
    model_config = ModelConfig(
        embedding_dim=50,
        num_classes=2
    )

    # Create classifier
    print("\n🔨 Creating classifier...")
    classifier = torchTextClassifiers(
        tokenizer=tokenizer,
        model_config=model_config
    )
    print("✅ Classifier created successfully!")
    
    # Train the model
    print("\n🎯 Training model...")
    training_config = TrainingConfig(
        num_epochs=20,
        batch_size=4,
        lr=1e-3,
        patience_early_stopping=5,
        num_workers=0  # Use 0 for simple examples to avoid multiprocessing issues
    )
    classifier.train(
        X_train, y_train, X_val, y_val,
        training_config=training_config,
        verbose=True
    )
    print("✅ Training completed!")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    result = classifier.predict(X_test)
    predictions = result["prediction"].squeeze().numpy()  # Extract predictions from dictionary
    confidence = result["confidence"].squeeze().numpy()  # Extract confidence scores
    print(f"Predictions: {predictions}")
    print(f"Confidence: {confidence}")
    print(f"True labels: {y_test}")

    # Calculate accuracy
    accuracy = (predictions == y_test).mean()
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