"""
Basic Text Classification Example

This example demonstrates how to use torchTextClassifiers for binary
text classification using the Wrapper.
"""

import os
import random
import warnings

import numpy as np
import torch
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer


def main():
    # Suppress PyTorch Lightning batch_size inference warnings for cleaner output
    warnings.filterwarnings(
        'ignore',
        message='.*',
        category=UserWarning,
        module='pytorch_lightning'
    )

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
        "Cheap plastic, feels like it will break soon.",
        "Absolutely fantastic! Exceeded all my expectations.",
        "Horrible experience. Customer service was rude and unhelpful.",
        "Best purchase I've made this year. Five stars!",
        "Defective item arrived. Packaging was also damaged.",
        "Super impressed with the performance and durability.",
        "Total disappointment. Doesn't match the description at all.",
        "Wonderful product! My whole family loves it.",
        "Avoid at all costs. Complete waste of time and money.",
        "Remarkable quality for the price. Very satisfied!",
        "Broke within a week. Clearly poor manufacturing.",
        "Exceptional value! Would definitely buy again.",
        "Misleading photos. Product looks nothing like advertised.",
        "Works like a charm. Installation was easy too.",
        "Returned it immediately. Not worth even half the price.",
        "Beautiful design and sturdy construction. Love it!",
        "Arrived late and damaged. Very frustrating experience.",
        "Top-notch quality! Highly recommend to everyone.",
        "Uncomfortable and poorly made. Regret buying this.",
        "Perfect fit and great finish. Couldn't be happier!",
        "Stopped working after two uses. Complete junk."
    ])

    y_train = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # 1=positive, 0=negative
    
    # Validation data
    X_val = np.array([
        "Good product, satisfied with purchase.",
        "Not worth the money, poor quality.",
        "Really happy with this purchase. Great item!",
        "Disappointed with the quality. Expected better.",
        "Solid product that does what it promises.",
        "Don't waste your money on this. Very poor.",
        "Impressive quality and quick delivery.",
        "Malfunctioned right out of the box. Terrible."
    ])
    y_val = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    
    # Test data
    X_test = np.array([
        "This is an amazing product with great features!",
        "Completely disappointed with this purchase.",
        "Excellent build quality and works as expected.",
        "Not recommended. Had issues from day one.",
        "Fantastic product! Worth every penny.",
        "Failed to meet basic expectations. Very poor.",
        "Love it! Exactly as described and high quality.",
        "Cheap materials and sloppy construction. Avoid.",
        "Superb performance and easy to use. Highly satisfied!",
        "Unreliable and frustrating. Should have bought elsewhere."
    ])
    y_test = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    
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
    print(classifier)
    # Train the model
    print("\n🎯 Training model...")
    training_config = TrainingConfig(
        num_epochs=20,
        batch_size=4,
        lr=1e-3,
        patience_early_stopping=5,
        num_workers=0,  # Use 0 for simple examples to avoid multiprocessing issues
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