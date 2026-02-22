import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        if not self.word_to_id:
            for i, token in enumerate([self.pad_token, self.unk_token, self.bos_token, self.eos_token]):
                self.word_to_id[token] = i
                self.id_to_word[i] = token

        index = len(self.word_to_id)

        for text in texts:
            for token in text.split():
                if token not in self.word_to_id:
                    self.word_to_id[token] = index
                    self.id_to_word[index] = token
                    index += 1
        
        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        return [self.word_to_id.get(word, self.word_to_id[self.unk_token])
                for word in text.split()]

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        return " ".join(self.id_to_word.get(i, self.unk_token) for i in ids)