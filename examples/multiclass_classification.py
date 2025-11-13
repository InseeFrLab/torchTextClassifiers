"""
Multi-class Text Classification Example

This example demonstrates multi-class text classification using
torchTextClassifiers for sentiment analysis with
3 classes: positive, negative, and neutral.
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

    print("🎭 Multi-class Text Classification Example")
    print("=" * 50)

    # Create multi-class sample data (3 classes: 0=negative, 1=neutral, 2=positive)
    print("📝 Creating multi-class sentiment data...")
    X_train = np.array([
        # Negative examples (class 0)
        "This product is terrible and I hate it completely.",
        "Worst purchase ever. Total waste of money.",
        "Absolutely awful quality. Very disappointed.",
        "Poor service and terrible product quality.",
        "I regret buying this. Complete failure.",
        
        # Neutral examples (class 1)  
        "The product is okay, nothing special though.",
        "It works but could be better designed.",
        "Average quality for the price point.",
        "Not bad but not great either.",
        "It's fine, meets basic expectations.",
        
        # Positive examples (class 2)
        "Excellent product! Highly recommended!",
        "Amazing quality and great customer service.",
        "Perfect! Exactly what I was looking for.",
        "Outstanding value and excellent performance.",
        "Love it! Will definitely buy again."
    ])
    
    y_train = np.array([0, 0, 0, 0, 0,  # negative
                       1, 1, 1, 1, 1,  # neutral  
                       2, 2, 2, 2, 2]) # positive
    
    # Validation data
    X_val = np.array([
        "Bad quality, not recommended.",     # negative
        "It's okay, does the job.",          # neutral
        "Great product, very satisfied!"     # positive
    ])
    y_val = np.array([0, 1, 2])
    
    # Test data
    X_test = np.array([
        "This is absolutely horrible!",
        "It's an average product, nothing more.",
        "Fantastic! Love every aspect of it!",
        "Really poor design and quality.",
        "Works well, good value for money.",
        "Outstanding product with amazing features!"
    ])
    y_test = np.array([0, 1, 2, 0, 1, 2])
    
    print(f"Training samples: {len(X_train)}")
    print(f"Class distribution: Negative={sum(y_train==0)}, Neutral={sum(y_train==1)}, Positive={sum(y_train==2)}")

    # Create and train tokenizer
    print("\n🏗️ Creating and training WordPiece tokenizer...")
    tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=128)
    training_corpus = X_train.tolist()
    tokenizer.train(training_corpus)
    print("✅ Tokenizer trained successfully!")

    # Create model configuration for 3 classes
    print("\n🔧 Creating model configuration...")
    model_config = ModelConfig(
        embedding_dim=64,
        num_classes=3  # 3 classes for sentiment (negative, neutral, positive)
    )

    # Create classifier
    print("\n🔨 Creating multi-class classifier...")
    classifier = torchTextClassifiers(
        tokenizer=tokenizer,
        model_config=model_config
    )
    print("✅ Classifier created successfully!")

    # Train the model
    print("\n🎯 Training model...")
    training_config = TrainingConfig(
        num_epochs=30,
        batch_size=8,
        lr=1e-3,
        patience_early_stopping=7,
        num_workers=0,
        trainer_params={'deterministic': True}
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
    predictions = result["prediction"].squeeze().numpy()
    print(f"Predictions: {predictions}")
    print(f"True labels: {y_test}")

    # Calculate accuracy
    accuracy = (predictions == y_test).mean()
    print(f"Test accuracy: {accuracy:.3f}")
    
    # Define class names for better output
    class_names = ["Negative", "Neutral", "Positive"]
    
    # Show detailed results
    print("\n📊 Detailed Results:")
    print("-" * 60)
    correct_predictions = 0
    for i, (text, pred, true) in enumerate(zip(X_test, predictions, y_test)):
        predicted_sentiment = class_names[pred]
        true_sentiment = class_names[true]
        correct = pred == true
        if correct:
            correct_predictions += 1
        status = "✅" if correct else "❌"
        
        print(f"{i+1}. {status} Predicted: {predicted_sentiment}, True: {true_sentiment}")
        print(f"   Text: {text}")
        print()
    
    print(f"Final Accuracy: {correct_predictions}/{len(X_test)} = {correct_predictions/len(X_test):.3f}")
  
   
    print("\n🎉 Multi-class example completed successfully!")

if __name__ == "__main__":
    main()