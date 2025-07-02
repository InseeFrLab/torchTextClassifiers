"""
Mixed Features Classification Example

This example demonstrates how to use torchTextClassifiers with both
text and categorical features for product review classification.
"""

import numpy as np
from torchTextClassifiers import create_fasttext

def main():
    print("🔀 Mixed Features Classification Example")
    print("=" * 50)
    
    # Create sample data with text + categorical features
    print("📝 Creating mixed feature data...")
    
    # Text reviews - expanded dataset
    reviews = np.array([
        "Great product, fast delivery!",
        "Poor quality, disappointing purchase.",
        "Excellent value for money.",
        "Terrible customer service experience.",
        "Amazing product, highly recommend!",
        "Not worth the price, poor quality.",
        "Perfect fit, great design!",
        "Defective item, had to return.",
        "Outstanding quality and service.",
        "Overpriced for what you get.",
        "Superb craftsmanship and attention to detail.",
        "Cheaply made, broke after one week of use.",
        "Exactly what I needed, fits perfectly.",
        "Arrived damaged and customer service was unhelpful.",
        "Best purchase I've made this year!",
        "Waste of money, completely useless product.",
        "Impressive build quality and modern design.",
        "False advertising, nothing like the pictures.",
        "Works flawlessly, exceeded my expectations.",
        "Returned it immediately, complete garbage.",
        "Beautiful packaging and premium feel.",
        "Cheap plastic construction, very flimsy.",
        "Fast shipping and excellent communication.",
        "Seller was rude and unresponsive to questions.",
        "Highly durable and well-engineered product.",
        "Broke on first use, total manufacturing defect.",
        "Stylish appearance and great functionality.",
        "Misleading description, not as advertised.",
        "Professional quality at reasonable price.",
        "Overpriced junk, save your money elsewhere.",
        "Innovative features and user-friendly design.",
        "Outdated technology, better alternatives available.",
        "Prompt delivery and secure packaging.",
        "Took forever to arrive and was poorly packed.",
        "Premium materials and exceptional workmanship.",
        "Feels cheap and looks even worse in person.",
        "Versatile product with many useful applications.",
        "Limited functionality, not worth the investment.",
        "Reliable performance and consistent quality.",
        "Constant malfunctions and technical issues.",
        "Elegant design that complements any space.",
        "Ugly appearance and doesn't match the photos.",
        "Easy to use with clear instructions included.",
        "Confusing setup process and poor documentation.",
        "Great customer support and hassle-free returns.",
        "Terrible warranty service and unhelpful staff.",
        "Compact size perfect for small spaces.",
        "Too bulky and takes up unnecessary room.",
        "Energy efficient and environmentally friendly.",
        "Power hungry device that increases electricity bills."
    ])
    
    # Categorical features: [product_category, price_range, brand_tier]
    # product_category: 0=Electronics, 1=Clothing, 2=Home
    # price_range: 0=Budget, 1=Mid-range, 2=Premium  
    # brand_tier: 0=Generic, 1=Known, 2=Premium
    categorical_features = np.array([
        [0, 2, 2],  # Electronics, Premium price, Premium brand
        [1, 0, 0],  # Clothing, Budget, Generic brand
        [2, 1, 1],  # Home, Mid-range, Known brand
        [0, 1, 0],  # Electronics, Mid-range, Generic brand
        [1, 2, 2],  # Clothing, Premium, Premium brand
        [2, 0, 0],  # Home, Budget, Generic brand
        [0, 1, 1],  # Electronics, Mid-range, Known brand
        [1, 0, 0],  # Clothing, Budget, Generic brand
        [2, 2, 2],  # Home, Premium, Premium brand
        [0, 2, 1],  # Electronics, Premium, Known brand
        [2, 2, 2],  # Home, Premium, Premium brand
        [1, 0, 0],  # Clothing, Budget, Generic brand
        [0, 1, 1],  # Electronics, Mid-range, Known brand
        [2, 0, 0],  # Home, Budget, Generic brand
        [1, 2, 2],  # Clothing, Premium, Premium brand
        [0, 0, 0],  # Electronics, Budget, Generic brand
        [2, 1, 1],  # Home, Mid-range, Known brand
        [1, 0, 0],  # Clothing, Budget, Generic brand
        [0, 2, 2],  # Electronics, Premium, Premium brand
        [2, 0, 0],  # Home, Budget, Generic brand
        [1, 2, 2],  # Clothing, Premium, Premium brand
        [0, 0, 0],  # Electronics, Budget, Generic brand
        [2, 1, 1],  # Home, Mid-range, Known brand
        [1, 0, 0],  # Clothing, Budget, Generic brand
        [0, 2, 2],  # Electronics, Premium, Premium brand
        [2, 1, 0],  # Home, Mid-range, Generic brand
        [1, 1, 1],  # Clothing, Mid-range, Known brand
        [0, 0, 0],  # Electronics, Budget, Generic brand
        [2, 2, 2],  # Home, Premium, Premium brand
        [1, 0, 0],  # Clothing, Budget, Generic brand
        [0, 2, 2],  # Electronics, Premium, Premium brand
        [2, 1, 1],  # Home, Mid-range, Known brand
        [1, 1, 1],  # Clothing, Mid-range, Known brand
        [0, 0, 0],  # Electronics, Budget, Generic brand
        [2, 2, 2],  # Home, Premium, Premium brand
        [1, 0, 0],  # Clothing, Budget, Generic brand
        [0, 1, 1],  # Electronics, Mid-range, Known brand
        [2, 0, 0],  # Home, Budget, Generic brand
        [1, 2, 2],  # Clothing, Premium, Premium brand
        [0, 1, 0],  # Electronics, Mid-range, Generic brand
        [2, 2, 2],  # Home, Premium, Premium brand
        [1, 0, 0],  # Clothing, Budget, Generic brand
        [0, 1, 1],  # Electronics, Mid-range, Known brand
        [2, 0, 0],  # Home, Budget, Generic brand
        [1, 2, 2],  # Clothing, Premium, Premium brand
        [0, 0, 0],  # Electronics, Budget, Generic brand
        [2, 1, 1],  # Home, Mid-range, Known brand
        [1, 1, 0],  # Clothing, Mid-range, Generic brand
        [0, 2, 2],  # Electronics, Premium, Premium brand
        [2, 0, 0],  # Home, Budget, Generic brand
    ])
    
    # Combine text and categorical features
    X_train = np.column_stack([reviews, categorical_features])
    
    # Labels: 1=positive review, 0=negative review
    y_train = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0,  # First 10 (original)
                        1, 0, 1, 0, 1, 0, 1, 0, 1, 0,  # 11-20
                        1, 0, 1, 0, 1, 0, 1, 0, 1, 0,  # 21-30
                        1, 0, 1, 0, 1, 0, 1, 0, 1, 0,  # 31-40
                        1, 0, 1, 0, 1, 0, 1, 0, 1, 0]) # 41-50
    
    # Validation data - expanded
    val_reviews = np.array([
        "Good product, satisfied with purchase.",
        "Not impressed, poor value for money.",
        "Exceptional quality and fast shipping.",
        "Defective product, poor manufacturing.",
        "Love the design and functionality.",
        "Completely useless, waste of money.",
        "Sturdy construction and reliable performance.",
        "Cheap materials, broke after few days.",
        "Perfect size and excellent features.",
        "Overpriced and underwhelming quality.",
        "Great customer service and support.",
        "Terrible experience, would not recommend.",
        "High-quality materials and craftsmanship.",
        "Poor durability, not worth the price.",
        "Sleek design and intuitive interface."
    ])
    val_categorical = np.array([
        [1, 1, 1],  # Clothing, Mid-range, Known brand
        [2, 0, 0],  # Home, Budget, Generic brand
        [0, 2, 2],  # Electronics, Premium, Premium brand
        [1, 0, 0],  # Clothing, Budget, Generic brand
        [2, 1, 1],  # Home, Mid-range, Known brand
        [0, 0, 0],  # Electronics, Budget, Generic brand
        [1, 2, 2],  # Clothing, Premium, Premium brand
        [2, 0, 0],  # Home, Budget, Generic brand
        [0, 1, 1],  # Electronics, Mid-range, Known brand
        [1, 1, 0],  # Clothing, Mid-range, Generic brand
        [2, 2, 2],  # Home, Premium, Premium brand
        [0, 0, 0],  # Electronics, Budget, Generic brand
        [1, 2, 2],  # Clothing, Premium, Premium brand
        [2, 0, 0],  # Home, Budget, Generic brand
        [0, 1, 1]   # Electronics, Mid-range, Known brand
    ])
    X_val = np.column_stack([val_reviews, val_categorical])
    y_val = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    # Test data - expanded
    test_reviews = np.array([
        "Fantastic product with excellent features!",
        "Complete waste of money, very poor quality.",
        "Great value and wonderful customer service.",
        "Outstanding build quality and durability.",
        "Flimsy construction, broke immediately.",
        "Impressive performance and reliability.",
        "Disappointing purchase, not recommended.",
        "Beautiful design and premium materials.",
        "Cheap knock-off, avoid at all costs.",
        "Exceeded expectations in every way.",
        "Faulty product with numerous defects.",
        "Highly recommended for its quality.",
        "Poorly designed and difficult to use.",
        "Excellent investment, worth every penny.",
        "Regret buying this inferior product.",
        "Innovative solution to common problems.",
        "Outdated design with limited features.",
        "Reliable and efficient performance.",
        "Constant issues and poor reliability.",
        "Perfect addition to any collection."
    ])
    test_categorical = np.array([
        [0, 2, 2],  # Electronics, Premium, Premium brand
        [1, 0, 0],  # Clothing, Budget, Generic brand  
        [2, 1, 1],  # Home, Mid-range, Known brand
        [0, 1, 1],  # Electronics, Mid-range, Known brand
        [1, 0, 0],  # Clothing, Budget, Generic brand
        [2, 2, 2],  # Home, Premium, Premium brand
        [0, 0, 0],  # Electronics, Budget, Generic brand
        [1, 2, 2],  # Clothing, Premium, Premium brand
        [2, 0, 0],  # Home, Budget, Generic brand
        [0, 2, 2],  # Electronics, Premium, Premium brand
        [1, 1, 0],  # Clothing, Mid-range, Generic brand
        [2, 1, 1],  # Home, Mid-range, Known brand
        [0, 0, 0],  # Electronics, Budget, Generic brand
        [1, 2, 2],  # Clothing, Premium, Premium brand
        [2, 0, 0],  # Home, Budget, Generic brand
        [0, 1, 1],  # Electronics, Mid-range, Known brand
        [1, 0, 0],  # Clothing, Budget, Generic brand
        [2, 2, 2],  # Home, Premium, Premium brand
        [0, 1, 0],  # Electronics, Mid-range, Generic brand
        [1, 1, 1]   # Clothing, Mid-range, Known brand
    ])
    X_test = np.column_stack([test_reviews, test_categorical])
    y_test = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    print(f"Training samples: {len(X_train)}")
    print(f"Feature dimensions: Text + {categorical_features.shape[1]} categorical features")
    print(f"Categorical feature ranges:")
    print(f"  - Product category: {np.unique(categorical_features[:, 0])}")
    print(f"  - Price range: {np.unique(categorical_features[:, 1])}")
    print(f"  - Brand tier: {np.unique(categorical_features[:, 2])}")
    
    # Create FastText classifier with categorical feature support
    print("\n🏗️ Creating FastText classifier with categorical features...")
    classifier = create_fasttext(
        embedding_dim=64,
        sparse=False,
        num_tokens=5000,
        min_count=1,
        min_n=3,
        max_n=6,
        len_word_ngrams=2,
        num_classes=2,
        # Categorical feature configuration
        categorical_vocabulary_sizes=[3, 3, 3],  # 3 categories each
        categorical_embedding_dims=[8, 8, 8]     # 8-dim embeddings each
    )
    
    # Build the model
    print("\n🔨 Building model with mixed features...")
    classifier.build(X_train, y_train)
    print("✅ Model built successfully!")
    print(f"✅ Categorical features detected and configured")
    
    # Train the model
    print("\n🎯 Training model...")
    classifier.train(
        X_train, y_train, X_val, y_val,
        num_epochs=25,
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
    
    # Show detailed results with feature analysis
    print("\n📊 Detailed Results:")
    print("-" * 80)
    category_names = ["Electronics", "Clothing", "Home"]
    price_names = ["Budget", "Mid-range", "Premium"]
    brand_names = ["Generic", "Known", "Premium"]
    
    for i, (text, cat_feats, pred, true) in enumerate(zip(test_reviews, test_categorical, predictions, y_test)):
        sentiment = "Positive" if pred == 1 else "Negative"
        correct = "✅" if pred == true else "❌"
        
        print(f"{i+1}. {correct} Predicted: {sentiment}")
        print(f"   Text: {text}")
        print(f"   Category: {category_names[cat_feats[0]]}")
        print(f"   Price: {price_names[cat_feats[1]]}")
        print(f"   Brand: {brand_names[cat_feats[2]]}")
        print()
    
    # Demonstrate feature importance by comparing with text-only model
    print("\n🔬 Comparing with text-only model...")
    text_only_classifier = create_fasttext(
        embedding_dim=64,
        sparse=False,
        num_tokens=5000,
        min_count=1,
        min_n=3,
        max_n=6,
        len_word_ngrams=2,
        num_classes=2
        # No categorical features
    )
    
    # Train text-only model
    text_only_classifier.build(reviews.reshape(-1, 1), y_train)
    text_only_classifier.train(
        reviews.reshape(-1, 1), y_train, 
        val_reviews.reshape(-1, 1), y_val,
        num_epochs=25,
        batch_size=4,
        patience_train=5,
        verbose=False
    )
    
    text_only_accuracy = text_only_classifier.validate(test_reviews.reshape(-1, 1), y_test)
    
    print(f"Mixed features accuracy: {accuracy:.3f}")
    print(f"Text-only accuracy: {text_only_accuracy:.3f}")
    print(f"Improvement: {accuracy - text_only_accuracy:.3f}")
    
    # Save configuration
    print("\n💾 Saving model configuration...")
    classifier.to_json('mixed_features_classifier_config.json')
    print("✅ Configuration saved to 'mixed_features_classifier_config.json'")
    
    print("\n🎉 Mixed features example completed successfully!")

if __name__ == "__main__":
    main()