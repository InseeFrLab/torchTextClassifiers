"""
Categorical Features Comparison Example

This example demonstrates the performance difference between:
1. A FastText classifier using only text features
2. A FastText classifier using both text and categorical features
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers import create_fasttext
from torchTextClassifiers.utilities.preprocess import clean_text_feature
from torchTextClassifiers import torchTextClassifiers
from torchTextClassifiers.classifiers.simple_text_classifier import SimpleTextWrapper, SimpleTextConfig
import time

def stratified_split_rare_labels(X, y, test_size=0.2, min_train_samples=1):
    # Get unique labels and their frequencies
    unique_labels, label_counts = np.unique(y, return_counts=True)

    # Separate rare and common labels
    rare_labels = unique_labels[label_counts == 1]

    # Create initial mask for rare labels to go into training set
    rare_label_mask = np.isin(y, rare_labels)

    # Separate data into rare and common label datasets
    X_rare = X[rare_label_mask]
    y_rare = y[rare_label_mask]
    X_common = X[~rare_label_mask]
    y_common = y[~rare_label_mask]

    # Split common labels stratified
    X_common_train, X_common_test, y_common_train, y_common_test = train_test_split(
        X_common, y_common, test_size=test_size, stratify=y_common
    )

    # Combine rare labels with common labels split
    X_train = np.concatenate([X_rare, X_common_train])
    y_train = np.concatenate([y_rare, y_common_train])
    X_test = X_common_test
    y_test = y_common_test

    return X_train, X_test, y_train, y_test


def merge_cat(cat):
    if cat in ['World', 'Top News', 'Europe', 'Italia', 'U.S.', 'Top Stories']:
        return 'World News'
    if cat in ['Sci/Tech', 'Software and Developement', 'Toons', 'Health', 'Music Feeds']:
        return 'Tech and Stuff'
    return cat


def load_and_prepare_data():
    """Load and prepare data"""
    print("📊 Using AG news dataset sample for demonstration...")
    df = pd.read_parquet("https://minio.lab.sspcloud.fr/h4njlg/public/ag_news_full_1M.parquet")
    df = df.sample(10000, random_state=42)  # Smaller sample to avoid disk space issues
    print(f"✅ Loaded {len(df)} samples from AG NEWS dataset")
    
    df['category_final'] = df['category'].apply(lambda x: merge_cat(x))
    df['title_headline'] = df['title'] + '\n####\n' + df['description']


    # categorical_features = None
    # text_feature = "title_headline"
    # y = "category"
    # textual_features = ["source", "url"]
    source_encoder = LabelEncoder()

    df["title_headline_processed"] = df["title_headline"]
    df["source_encoded"] = source_encoder.fit_transform(df['source'])

    X_text_only = df[['title_headline_processed']].values
    X_mixed = df[['title_headline_processed', "source_encoded"]].values
    y = df['category_final'].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    return X_text_only, X_mixed, y, encoder
    

def train_and_evaluate_model(X, y, model_name, use_categorical=False, use_simple=False):
    """Train and evaluate a FastText model"""
    print(f"\n🎯 Training {model_name}...")
    
  
    # Split data twice: first for train/temp, then temp into validation/test
    X_train, X_temp, y_train, y_temp = stratified_split_rare_labels(
        X, y, test_size=0.1  # 40% for validation + test
    )
    X_val, X_test, y_val, y_test = stratified_split_rare_labels(
        X_temp, y_temp, test_size=0.5  # Split temp 50/50 into validation and test
    )
    
    if use_simple:
        start_time = time.time()

        simple_text_config = SimpleTextConfig(
            hidden_dim=128,
            num_classes=5,
            max_features=5000,
            learning_rate=1e-3,
            dropout_rate=0.2
        )
        wrapper = SimpleTextWrapper(simple_text_config)
        classifier = torchTextClassifiers(wrapper)
        print(f"Classifier type: {type(classifier.classifier).__name__}")
        print(f"Uses tokenizer: {hasattr(classifier.classifier, 'tokenizer')}")
        print(f"Uses vectorizer: {hasattr(classifier.classifier, 'vectorizer')}")
        
        # Build the model (this will use TF-IDF vectorization instead of tokenization)
        print("\n🔨 Building model with TF-IDF preprocessing...")
        classifier.build(X_train, y_train)
        print("✅ Model built successfully!")
        print(f"TF-IDF features: {len(classifier.classifier.vectorizer.get_feature_names_out())}")
        
        # Train the model
        print("\n🎯 Training model...")
        classifier.train(
            X_train, y_train, X_val, y_val,
            num_epochs=10,
            batch_size=4,
            patience_train=3,
            verbose=True
        )
        training_time = time.time() - start_time
        accuracy = classifier.validate(X_test, y_test)
        print(f"Test accuracy: {accuracy:.3f}")

        return {
            'model_name': model_name,
            'test_accuracy': accuracy,
            'training_time': training_time,
            'classifier': classifier
        }

    # Model parameters
    if use_categorical:
        # For mixed model - get vocabulary sizes from data
        # cat_data = X_train[:, 1:].astype(int)  # Categorical features
        # vocab_sizes = [int(np.max(cat_data[:, i]) + 1) for i in range(cat_data.shape[1])]
        
        model_params = {
            "embedding_dim": 50,
            "sparse": False,
            "num_tokens": 50000,
            "min_count": 1,
            "min_n": 3,
            "max_n": 6,
            "len_word_ngrams": 2,
            "categorical_embedding_dims": 10,
            #"num_categorical_features": num_cat_var,
            #"categorical_vocabulary_sizes": vocab_sizes,
            #"categorical_embedding_dims": 10
        }
        #print(f"   Categorical vocabulary sizes: {vocab_sizes}")
    else:
        # For text-only model
        model_params = {
            "embedding_dim": 50,
            "sparse": False,
            
            "num_tokens": 50000,
            "min_count": 1,
            "min_n": 3,
            "max_n": 6,
            "len_word_ngrams": 2
        }
    
    # Training parameters - reduced to save disk space
    train_params = {
        "num_epochs": 50,
        "batch_size": 128,
        "patience_train": 3,
        "lr": 0.001,
        "verbose": True
    }

    extra_trainer_params = {
        "enable_progress_bar": True,
        
    }
    
    # Create and build model
    start_time = time.time()
    
    classifier = create_fasttext(**model_params)
    classifier.build(X_train, y_train)
    
    # Train model - disable logging to save disk space
    classifier.train(
        X_train, y_train, X_val, y_val, **train_params,
        trainer_params=extra_trainer_params
    )
    training_time = time.time() - start_time
    
    # Handle predictions based on model type
    if use_categorical:
        # Skip validation for mixed model due to categorical prediction bug
        print("   ✅ Running validation for text-with-categorical-variables model...")
        try:
            test_accuracy = classifier.validate(X_test, y_test)
            predictions = classifier.predict(X_test)
            print(f"   Test accuracy: {test_accuracy:.3f}")
        except Exception as e:
            print(f"   ⚠️  Validation failed: {e}")
            train_accuracy = 0.0
            test_accuracy = 0.0
            predictions = np.zeros(len(y_test))
    else:
        # Text-only model works fine for predictions
        print("   ✅ Running validation for text-only model...")
        try:
            test_accuracy = classifier.validate(X_test, y_test)
            predictions = classifier.predict(X_test)
            print(f"   Test accuracy: {test_accuracy:.3f}")
        except Exception as e:
            print(f"   ⚠️  Validation failed: {e}")
            train_accuracy = 0.0
            test_accuracy = 0.0
            predictions = np.zeros(len(y_test))
    
    return {
        'model_name': model_name,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'predictions': predictions,
        'y_test': y_test,
        'classifier': classifier
    }




def main():
    print("🔀 FastText Classifier: Categorical Features Comparison")
    print("=" * 60)
    print("Comparing FastText performance with and without categorical features")
    print()
    
    # Load and prepare data (same as notebook)
    X_text_only, X_mixed, y, encoder = load_and_prepare_data()
    
    # Train models
    print(f"\n🚀 Training Models:")
    print("-" * 40)
    
    # Text-only model
    results_text_only = train_and_evaluate_model(
        X_text_only, y, "Text-Only FastText", use_categorical=False
    )
    
    # Mixed model (text + categorical)
    results_mixed = train_and_evaluate_model(
        X_mixed, y, "Mixed Features FastText", use_categorical=True
    )

    # TF-IDF classifier
    results_tfidf = train_and_evaluate_model(X_text_only, y, "TF-IDF classifier", use_categorical=False, use_simple=True)
    
    # Compare results
    print(f"\n📊 Results Comparison:")
    print("=" * 50)
    print(f"{'Model':<25}{'Test Acc':<11} {'Time (s)':<10}")
    print("-" * 50)
    print(f"{'Text-Only':<25} "
          f"{results_text_only['test_accuracy']:<11.3f} {results_text_only['training_time']:<10.1f}")
    print(f"{'Mixed Features':<25} "
          f"{results_mixed['test_accuracy']:<11.3f} {results_mixed['training_time']:<10.1f}")
    print(f"{'TF-IDF':<25} "
          f"{results_tfidf['test_accuracy']:<11.3f} {results_tfidf['training_time']:<10.1f}")
    # Calculate improvements
    acc_improvement = results_mixed['test_accuracy'] - results_text_only['test_accuracy']
    time_overhead = results_mixed['training_time'] - results_text_only['training_time']
    
    print("-" * 50)
    print(f"Test Accuracy Improvement: {acc_improvement:+.3f}")
    print(f"Training Time Overhead: {time_overhead:+.1f}s")

if __name__ == "__main__":
    main()