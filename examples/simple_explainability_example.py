"""
Simple Explainability Example with ASCII Visualization
"""

import numpy as np
import sys
from torchTextClassifiers import create_fasttext


def main():
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
    
    # Create classifier
    classifier = create_fasttext(
        embedding_dim=50,
        sparse=False,
        num_tokens=1000,
        min_count=1,
        min_n=3,
        max_n=6,
        len_word_ngrams=2,
        num_classes=2,
        direct_bagging=False  # Required for explainability
    )
    
    # Train
    classifier.build(X_train, y_train)
    classifier.train(X_train, y_train, X_val, y_val, num_epochs=25, batch_size=8, verbose=False)
    
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
        
        # Get prediction
        prediction = classifier.predict(np.array([test_text]))[0]
        print(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
        
        # Get explainability scores
        try:
            pred, confidence, all_scores, all_scores_letters = classifier.predict_and_explain(np.array([test_text]))
            
            # Create ASCII histogram
            if all_scores is not None and len(all_scores) > 0:
                scores_data = all_scores[0][0]
                if hasattr(scores_data, 'tolist'):
                    scores = scores_data.tolist()
                else:
                    scores = [float(scores_data)]
                
                words = test_text.split()
                
                if len(words) == len(scores):
                    print("\n📊 Word Contribution Histogram:")
                    print("-" * 50)
                    
                    # Find max score for scaling
                    max_score = max(scores) if scores else 1
                    bar_width = 30  # max bar width in characters
                    
                    for word, score in zip(words, scores):
                        # Calculate bar length
                        bar_length = int((score / max_score) * bar_width)
                        bar = "█" * bar_length
                        
                        # Format output
                        print(f"{word:>12} | {bar:<30} {score:.4f}")
                    
                    print("-" * 50)
                else:
                    print(f"⚠️  Word/score mismatch: {len(words)} words vs {len(scores)} scores")
            else:
                print("⚠️  No explainability scores available")
                
        except Exception as e:
            print(f"⚠️  Explainability failed: {e}")
        
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
                
                # Get prediction
                prediction = classifier.predict(np.array([user_text]))[0]
                sentiment = "Positive" if prediction == 1 else "Negative"
                print(f"🎯 Prediction: {sentiment}")
                
                # Get explainability scores
                try:
                    pred, confidence, all_scores, all_scores_letters = classifier.predict_and_explain(np.array([user_text]))
                    
                    # Create ASCII histogram
                    if all_scores is not None and len(all_scores) > 0:
                        scores_data = all_scores[0][0]
                        if hasattr(scores_data, 'tolist'):
                            scores = scores_data.tolist()
                        else:
                            scores = [float(scores_data)]
                        
                        words = user_text.split()
                        
                        if len(words) == len(scores):
                            print("\n📊 Word Contribution Histogram:")
                            print("-" * 50)
                            
                            # Find max score for scaling
                            max_score = max(scores) if scores else 1
                            bar_width = 30  # max bar width in characters
                            
                            for word, score in zip(words, scores):
                                # Calculate bar length
                                bar_length = int((score / max_score) * bar_width)
                                bar = "█" * bar_length
                                
                                # Format output
                                print(f"{word:>12} | {bar:<30} {score:.4f}")
                            
                            print("-" * 50)
                            
                            # Show interpretation
                            top_word = max(zip(words, scores), key=lambda x: x[1])
                            print(f"💡 Most influential word: '{top_word[0]}' (score: {top_word[1]:.4f})")
                            
                        else:
                            print(f"⚠️  Word/score mismatch: {len(words)} words vs {len(scores)} scores")
                    else:
                        print("⚠️  No explainability scores available")
                        
                except Exception as e:
                    print(f"⚠️  Explainability failed: {e}")
                    print("🔍 Prediction available, but detailed explanation unavailable.")
                
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