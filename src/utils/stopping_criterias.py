from typing import List

import torch
from transformers import StoppingCriteria


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stopwords: List[str]):
        self.tokenizer = tokenizer
        self.stopwords = tuple(stopwords)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        output = self.tokenizer.decode(input_ids[0])
        if output.endswith(self.stopwords):
            return True
        return False
