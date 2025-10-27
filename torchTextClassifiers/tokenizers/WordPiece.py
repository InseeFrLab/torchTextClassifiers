import logging
import os
from typing import List

from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

from torchTextClassifiers.tokenizers import HuggingFaceTokenizer

logger = logging.getLogger(__name__)


class WordPieceTokenizer(HuggingFaceTokenizer):
    def __init__(self, vocab_size: int, trained: bool = False):
        """Largely inspired by https://huggingface.co/learn/llm-course/chapter6/8"""

        super().__init__(vocab_size)

        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.special_tokens = [
            self.unk_token,
            self.pad_token,
            self.cls_token,
            self.sep_token,
        ]
        self.vocab_size = vocab_size

        self.tokenizer = Tokenizer(models.WordPiece(unk_token=self.unk_token))

        self.tokenizer.normalizer = normalizers.BertNormalizer(
            lowercase=True
        )  # NFD, lowercase, strip accents - BERT style

        self.tokenizer.pre_tokenizer = (
            pre_tokenizers.BertPreTokenizer()
        )  # split on whitespace and punctuation - BERT style
        self.trained = trained

    def _post_training(self):
        if not self.trained:
            raise RuntimeError(
                "Tokenizer must be trained before applying post-training configurations."
            )

        self.tokenizer.post_processor = processors.BertProcessing(
            (self.cls_token, self.tokenizer.token_to_id(self.cls_token)),
            (self.sep_token, self.tokenizer.token_to_id(self.sep_token)),
        )
        self.tokenizer.decoder = decoders.WordPiece(prefix="##")
        self.padding_idx = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(pad_id=self.padding_idx, pad_token="[PAD]")

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)

    def train(
        self, training_corpus: List[str], save_path: str = None, filesystem=None, s3_save_path=None
    ):
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
        )
        self.tokenizer.train_from_iterator(training_corpus, trainer=trainer)
        self.trained = True
        self._post_training()

        if save_path:
            self.tokenizer.save(save_path)
            logger.info(f"💾 Tokenizer saved at {save_path}")
            if filesystem and s3_save_path:
                parent_dir = os.path.dirname(save_path)
                if not filesystem.exists(parent_dir):
                    filesystem.mkdirs(parent_dir)
                filesystem.put(save_path, s3_save_path)
                logger.info(f"💾 Tokenizer uploaded to S3 at {s3_save_path}")

    @classmethod
    def load(cls, load_path: str):
        loaded_tokenizer = PreTrainedTokenizerFast(tokenizer_file=load_path)
        instance = cls(vocab_size=len(loaded_tokenizer), trained=True)
        instance.tokenizer = loaded_tokenizer
        instance._post_training()
        return instance

    @classmethod
    def load_from_s3(cls, s3_path: str, filesystem):
        if filesystem.exists(s3_path) is False:
            raise FileNotFoundError(
                f"Tokenizer not found at {s3_path}. Please train it first (see src/train_tokenizers)."
            )

        with filesystem.open(s3_path, "rb") as f:
            json_str = f.read().decode("utf-8")

        tokenizer_obj = Tokenizer.from_str(json_str)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
        instance = cls(vocab_size=len(tokenizer), trained=True)
        instance.tokenizer = tokenizer
        instance._post_training()
        return instance
