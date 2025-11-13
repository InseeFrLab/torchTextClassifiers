"""
Simple Explainability Example with ASCII Visualization
"""

import os
import sys
import warnings

import numpy as np
import torch
from pytorch_lightning import seed_everything

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer
from torchTextClassifiers.utilities.plot_explainability import (
    map_attributions_to_char,
    map_attributions_to_word,
)


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

    print("🔍 Simple Explainability Example")

    # Enhanced training data with more diverse examples
    X_train = np.array([
        # Positive examples
        "I love this product",
        "Great quality and excellent service", 
        "Amazing design and fantastic performance",
        "Outstanding value for money",
        "Excellent customer support team",
        "Love the innovative features",
        "Perfect solution for my needs",
        "Highly recommend this item",
        "Superb build quality",
        "Wonderful experience overall",
        "Great value and fast delivery",
        "Excellent product with amazing results",
        "Love this fantastic design",
        "Perfect quality and great price",
        "Amazing customer service experience",
        
        # Negative examples  
        "This is terrible quality",
        "Poor design and cheap materials",
        "Awful experience with this product",
        "Terrible customer service response", 
        "Completely disappointing purchase",
        "Poor quality and overpriced item",
        "Awful build quality issues",
        "Terrible value for money",
        "Disappointing performance results",
        "Poor service and bad experience",
        "Awful design and cheap feel",
        "Terrible product with many issues",
        "Disappointing quality and poor value",
        "Bad experience with customer support",
        "Poor construction and awful materials"
    ])
    
    y_train = np.array([
        # Positive labels (1)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # Negative labels (0) 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])
    
    X_val = np.array([
        "Good product with decent quality",
        "Bad quality and poor service",
        "Excellent value and great design",
        "Terrible experience and awful quality"
    ])
    y_val = np.array([1, 0, 1, 0])

    # Create and train tokenizer
    print("\n🏗️ Creating and training WordPiece tokenizer...")
    tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=128)
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
        num_epochs=25,
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
    print("✅ Training completed!")
    
    # Test examples with different sentiments
    test_texts = [
        "This product is amazing!",
        "Poor quality and terrible service",
        "Great value for money",
        "Completely disappointing and awful experience",
        "Love this excellent design"
    ]
    
    print(f"\n🔍 Testing explainability on {len(test_texts)} examples:")
    print("=" * 60)
    
    for i, test_text in enumerate(test_texts, 1):
        print(f"\n📝 Example {i}:")
        print(f"Text: '{test_text}'")

        # Get prediction with explainability
        try:
            result = classifier.predict(np.array([test_text]), top_k=1, explain=True)

            # Extract prediction
            prediction = result["prediction"][0][0].item()
            confidence = result["confidence"][0][0].item()
            print(f"Prediction: {'Positive' if prediction == 1 else 'Negative'} (confidence: {confidence:.4f})")

            # Extract attributions and mapping info
            attributions = result["attributions"][0][0]  # shape: (seq_len,)
            offset_mapping = result["offset_mapping"][0]  # List of (start, end) tuples
            word_ids = result["word_ids"][0]  # List of word IDs for each token

            # Map token-level attributions to character-level (for ASCII visualization)
            char_attributions = map_attributions_to_char(
                attributions.unsqueeze(0),  # Add batch dimension: (1, seq_len)
                offset_mapping,
                test_text
            )[0]  # Get first result

            print("\n📊 Character-Level Contribution Visualization:")
            print("-" * 60)

            # Create a simple ASCII visualization by character
            max_attr = max(char_attributions) if len(char_attributions) > 0 else 1
            bar_width = 40

            # Group characters into words for better readability
            words = test_text.split()
            char_idx = 0

            for word in words:
                word_len = len(word)
                # Get attributions for this word
                word_attrs = char_attributions[char_idx:char_idx + word_len]
                if len(word_attrs) > 0:
                    avg_attr = sum(word_attrs) / len(word_attrs)
                    bar_length = int((avg_attr / max_attr) * bar_width) if max_attr > 0 else 0
                    bar = "█" * bar_length
                    print(f"{word:>15} | {bar:<40} {avg_attr:.4f}")
                char_idx += word_len + 1  # +1 for space

            print("-" * 60)

            # Show top contributing word
            char_idx = 0
            word_scores = []
            for word in words:
                word_len = len(word)
                word_attrs = char_attributions[char_idx:char_idx + word_len]
                if len(word_attrs) > 0:
                    word_scores.append((word, sum(word_attrs) / len(word_attrs)))
                char_idx += word_len + 1

            if word_scores:
                top_word, top_score = max(word_scores, key=lambda x: x[1])
                print(f"💡 Most influential word: '{top_word}' (avg score: {top_score:.4f})")

        except Exception as e:
            print(f"⚠️  Explainability failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Analysis completed for this example
        print(f"✅ Analysis completed for example {i}")
    
    print(f"\n🎉 Explainability analysis completed for {len(test_texts)} examples!")
    
    # Interactive section for user input (only if --interactive flag is provided)
    if "--interactive" in sys.argv:
        print("\n" + "="*60)
        print("🎯 Interactive Explainability Mode")
        print("="*60)
        print("Enter your own text to see predictions and explanations!")
        print("Type 'quit' or 'exit' to end the session.\n")
        
        while True:
            try:
                user_text = input("💬 Enter text: ").strip()
                
                if user_text.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thanks for using the explainability tool!")
                    break
                    
                if not user_text:
                    print("⚠️  Please enter some text.")
                    continue
                
                print(f"\n🔍 Analyzing: '{user_text}'")

                # Get prediction with explainability
                try:
                    result = classifier.predict(np.array([user_text]), top_k=1, explain=True)

                    # Extract prediction
                    prediction = result["prediction"][0][0].item()
                    confidence = result["confidence"][0][0].item()
                    sentiment = "Positive" if prediction == 1 else "Negative"
                    print(f"🎯 Prediction: {sentiment} (confidence: {confidence:.4f})")

                    # Extract attributions and mapping info
                    attributions = result["attributions"][0][0]  # shape: (seq_len,)
                    offset_mapping = result["offset_mapping"][0]  # List of (start, end) tuples
                    word_ids = result["word_ids"][0]  # List of word IDs for each token

                    # Map token-level attributions to character-level (for ASCII visualization)
                    char_attributions = map_attributions_to_char(
                        attributions.unsqueeze(0),  # Add batch dimension: (1, seq_len)
                        offset_mapping,
                        user_text
                    )[0]  # Get first result

                    print("\n📊 Character-Level Contribution Visualization:")
                    print("-" * 60)

                    # Create a simple ASCII visualization by character
                    max_attr = max(char_attributions) if len(char_attributions) > 0 else 1
                    bar_width = 40

                    # Group characters into words for better readability
                    words = user_text.split()
                    char_idx = 0

                    for word in words:
                        word_len = len(word)
                        # Get attributions for this word
                        word_attrs = char_attributions[char_idx:char_idx + word_len]
                        if len(word_attrs) > 0:
                            avg_attr = sum(word_attrs) / len(word_attrs)
                            bar_length = int((avg_attr / max_attr) * bar_width) if max_attr > 0 else 0
                            bar = "█" * bar_length
                            print(f"{word:>15} | {bar:<40} {avg_attr:.4f}")
                        char_idx += word_len + 1  # +1 for space

                    print("-" * 60)

                    # Show interpretation
                    char_idx = 0
                    word_scores = []
                    for word in words:
                        word_len = len(word)
                        word_attrs = char_attributions[char_idx:char_idx + word_len]
                        if len(word_attrs) > 0:
                            word_scores.append((word, sum(word_attrs) / len(word_attrs)))
                        char_idx += word_len + 1

                    if word_scores:
                        top_word, top_score = max(word_scores, key=lambda x: x[1])
                        print(f"💡 Most influential word: '{top_word}' (avg score: {top_score:.4f})")

                except Exception as e:
                    print(f"⚠️  Explainability failed: {e}")
                    print("🔍 Prediction available, but detailed explanation unavailable.")
                    import traceback
                    traceback.print_exc()
                
                print("\n" + "-"*50)
                
            except KeyboardInterrupt:
                print("\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"⚠️  Error: {e}")
                continue
    else:
        print("\n💡 Tip: Use --interactive flag to enter interactive mode for custom text analysis!")
        print("   Example: uv run python examples/simple_explainability_example.py --interactive")


if __name__ == "__main__":
    main()