from datasets import load_dataset

datasets = load_dataset('ag_news')
X_train = load_dataset('ag_news', split='train')

print(X_train)


