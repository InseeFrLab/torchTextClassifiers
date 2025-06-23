"""
Multi-class Text Classification Example

This example demonstrates multi-class text classification using
torchTextClassifiers with FastText for sentiment analysis with
3 classes: positive, negative, and neutral.
"""

import numpy as np
from torchTextClassifiers import create_fasttext

def main():
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
    
    # Create FastText classifier for 3 classes
    print("\n🏗️ Creating multi-class FastText classifier...")
    classifier = create_fasttext(
        embedding_dim=64,
        sparse=False,
        num_tokens=8000,
        min_count=1,
        min_n=3,
        max_n=6,
        len_word_ngrams=2,
        num_classes=3  # 3 classes for sentiment
    )
    
    # Build the model
    print("\n🔨 Building model...")
    classifier.build(X_train, y_train)
    print("✅ Model built successfully!")
    
    # Train the model
    print("\n🎯 Training model...")
    classifier.train(
        X_train, y_train, X_val, y_val,
        num_epochs=30,
        batch_size=8,
        patience_train=7,
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
    
    # Save configuration
    print("💾 Saving model configuration...")
    classifier.to_json('multiclass_classifier_config.json')
    print("✅ Configuration saved to 'multiclass_classifier_config.json'")
    
    # Demonstrate loading configuration
    print("\n🔄 Demonstrating configuration loading...")
    from torchTextClassifiers import torchTextClassifiers
    loaded_classifier = torchTextClassifiers.from_json('multiclass_classifier_config.json')
    print("✅ Configuration loaded successfully!")
    print(f"Loaded classifier type: {loaded_classifier.classifier_type}")
    print(f"Loaded num_classes: {loaded_classifier.config.num_classes}")
    
    print("\n🎉 Multi-class example completed successfully!")

if __name__ == "__main__":
    main()