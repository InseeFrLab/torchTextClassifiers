from datasets import load_dataset
import os
import random
import warnings

import numpy as np
import torch
from pytorch_lightning import seed_everything

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer

datasets = load_dataset('ag_news')


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
    print("📝 Creating multi-class data...")
    train_dataset = datasets['train']
    X_train = np.array(train_dataset['text'], dtype=object)
    y_train = np.array(train_dataset['label'], dtype=np.int64)

    print(f"✓ X_train shape: {X_train.shape}, type: {type(X_train)}")
    print(f"✓ y_train shape: {y_train.shape}, type: {type(y_train)}")
    print(f"✓ Premier exemple: {X_train[0][:100]}...")

    # Validation data
    test_dataset = datasets['test']
    X_val = np.array(test_dataset['text'], dtype=object)
    y_val = np.array(test_dataset['label'], dtype=np.int64)


    # Create and train tokenizer
    print("\n🏗️ Creating and training WordPiece tokenizer...")
    tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=128)
    tokenizer.train(X_train)
    print("✅ Tokenizer trained successfully!")

    # Create model configuration for 3 classes
    print("\n🔧 Creating model configuration...")
    model_config = ModelConfig(
        embedding_dim=64,
        num_classes=4  
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
    class_names = ["0", "1", "2", "3"]
    
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