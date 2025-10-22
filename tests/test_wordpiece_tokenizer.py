from torchTextClassifiers.tokenizers import WordPieceTokenizer


class TestWordPieceTokenizer:
    def test_init(self):
        tokenizer = WordPieceTokenizer(1000)
        assert tokenizer is not None

    def test_train(self, sample_text_data):
        tokenizer = WordPieceTokenizer(1000)
        tokenizer.train(sample_text_data)
