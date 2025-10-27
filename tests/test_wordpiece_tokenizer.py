from torchTextClassifiers.tokenizers import WordPieceTokenizer


class TestWordPieceTokenizer:
    def test_init(self):
        tokenizer = WordPieceTokenizer(1000)
        assert tokenizer is not None

    def test_train(self, sample_text_data):
        tokenizer = WordPieceTokenizer(1000)
        tokenizer.train(sample_text_data)

    def test_tokenize(self, sample_text_data):
        tokenizer = WordPieceTokenizer(1000)
        tokenizer.train(sample_text_data)
        tokens = tokenizer.tokenize(sample_text_data[0])
        tokens = tokenizer.tokenize(list(sample_text_data))

        assert tokens is not None
