import torch
from torch.utils.data import Dataset

with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()
char_set = set(text)  # set of characters that appear in the text
char_size = len(char_set)

# We use one-hot encoding for character.
# So, it is useful to have dictionaries for converting characters to index and vice versa.
char2idx = {char: idx for idx, char in enumerate(char_set)}
idx2char = {idx: char for idx, char in enumerate(char_set)}


def word2t(word, char_size):
    """
    convert word to tensor, where each character in the word is 
    converted into one-hot vector. 
    Note that the output has the shape (word_len, 1, char_size).
    The shape 1 is there because torch expects the second index to be the channel index.

    Args:
        word(str): word to be converted into tensor (one-hot encoding)
        char_size(int): number of characters in the text

    Returns:
        torch.tensor: tensor form of word
    """
    word_len = len(word)
    word_t = torch.zeros(word_len, char_size)
    for i in range(word_len):
        idx = char2idx[word[i]]
        word_t[i, idx] = 1
    return word_t


def word2idx_t(word):
    """
    Converts word to tensor of indices (which is used for one-hot encoding)
    Note that the output tensor has the shape (word_len, 1)

    Args:
        word(str): word to be converted into tensor (tensor of indices)

    Returns:
        torch.tensor: tensor form of word.
    """
    word_len = len(word)
    idx_t = torch.zeros((word_len), dtype=torch.long)
    for i in range(word_len):
        idx_t[i] = char2idx[word[i]]
    return idx_t


class textDataset(Dataset):
    def __init__(self,
                 source_txt,
                 batch_size=32,
                 ):
        self.batch_size = batch_size
        self.source_txt = source_txt
        self.len = len(source_txt) // 32 - 1  # throw away last bit of text so that __getitem__ does not cause any problem for the last iteration.

    def __len__(self,):
        return self.len

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        input_str = self.source_txt[start_idx:start_idx + self.batch_size]
        output_str = self.source_txt[start_idx + 1:start_idx + self.batch_size + 1]
        input_t = word2t(input_str, char_size)
        target_t = word2idx_t(output_str)
        return input_t, target_t
