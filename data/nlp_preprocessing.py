import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        splitted_pos = [sentence.split(' ') for sentence in positive]
        splitted_neg = [sentence.split(' ') for sentence in negative]
        splitted_all = splitted_pos + splitted_neg
        all_words = set([word for sentence in splitted_all for word in sentence])
        word2id = {word:id+1 for id, word in enumerate(sorted(all_words))}

        all_tensors = [torch.tensor([word2id[word] for word in sentence]).float() for sentence in splitted_all]
        
        return nn.utils.rnn.pad_sequence(all_tensors, batch_first=True)



