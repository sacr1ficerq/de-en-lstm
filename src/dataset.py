from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm

import re

unk_idx, pad_idx, bos_idx, eos_idx, num_idx = 0, 1, 2, 3, 4

def tokenize(line):
    return line.strip().split()

specials = ['<UNK>', '<PAD>', '<BOS>', '<EOS>', '<NUM>']


class Tokenizer():
    def __init__(self, filename, tokenize=tokenize):
        lines = []
        with open(filename, encoding='utf-8') as f:
            lines = f.readlines()
        sentences = [tokenize(line) for line in lines]
        self.sentences = sentences

pattern_float = r"^-?\+?\d+\.?\,?\d*\+?-?$"
# pattern_float = r"^-?\+?\d+\.?\,?\d*$"

class Vocab():
    def __init__(self, filename, min_freq=2, specials=specials):
        sentences = Tokenizer(filename).sentences
        self.last_num = None

        self.word_counter = Counter([word for words in sentences for word in words])
        self.vocab = {}

        self.specials = set(specials)

        vocab_set = set(self.word_counter.keys())

        self.all_words = []
        for word in vocab_set:
            if re.match(pattern_float, word) or self.word_counter[word] < min_freq: continue
            self.all_words.append(word)
        self.all_words = specials + sorted(set(self.all_words) - self.specials)

        for i, word in enumerate(self.all_words):
            self.vocab[word] = i

    def plot_vocab(self):
        cnt = Counter(self.word_counter.values())
        xs = sorted(cnt.keys())
        ys = [cnt[k] for k in xs]
        plt.xlabel('frequency')
        plt.ylabel('amount of frequency')
        plt.plot(xs[:100], ys[:100])
        plt.show()
        
        print((cnt[1]+2* cnt[2])/sum((np.array(list(cnt.values())))*(np.array(list(cnt.keys())))))

    def encode_word(self, word) -> int:
        if re.match(pattern_float, word):
            return num_idx
        return self.vocab[word] if word in self.vocab else self.vocab['<UNK>']

    def decode_idx(self, idx, src=None) -> str:
        if idx == num_idx:
            return '<NUM>'
        return self.all_words[idx] if idx < len(self) else '<UNK>'

    def encode(self, words, max_len=None):
        if words[0] != '<BOS>':
            words.insert(0, '<BOS>')
        if words[-1] != '<EOS>':
            words.append('<EOS>')
        
        if max_len:
            words = words[:max_len]
            if words[-1] != '<EOS>':
                words[-1] = ('<EOS>')

        return [self.encode_word(word) for word in words]

    def decode(self, idxs, ignore=set(['<UNK>', '<BOS>', '<EOS>', '<PAD>']), src=None):
        result = []
        has_num = False
        nums_idxs = []
        for i, idx in enumerate(idxs):
            if idx == eos_idx:
                break
            if idx == num_idx:
                has_num = True
                nums_idxs.append(i)
            if i != 0 and idxs[i-1] == idxs[i]: continue 
            result.append(self.decode_idx(idx))
        if has_num and src != None:
            nums_src = list(filter(lambda x: re.match(pattern_float, x), src))
            # print(nums_src)
            for t, idx in enumerate(nums_idxs):
                if len(nums_src) <= t:
                    break
                result[idx] = nums_src[t]
        return list(filter(lambda word: not word in ignore, result))

    def __len__(self):
        return len(self.all_words)


class TranslationDataset(Dataset):
    def __init__(self, vocab_src, vocab_trg, filename_src, filename_trg, max_len=48, device='cuda', sort_lengths=False):
        self.max_len = max_len
        word_sentences_src = Tokenizer(filename_src).sentences
        word_sentences_trg = Tokenizer(filename_trg).sentences

        lengths_src = [len(sentence) for sentence in word_sentences_src]
        lengths_trg = [len(sentence) for sentence in word_sentences_trg]
        
        self.cnt_src = Counter(lengths_src)
        self.cnt_trg = Counter(lengths_trg)

        sentences_src = []
        sentences_trg = []

        if sort_lengths:
            word_sentences_src, word_sentences_trg = zip(*sorted(zip(word_sentences_src, word_sentences_trg), 
                                                                 key=lambda x: len(x[0]),
                                                                 reverse=True))
            word_sentences_src = list(word_sentences_src)
            word_sentences_trg = list(word_sentences_trg)


        for i in tqdm(range(len(word_sentences_src))):
            new_src = vocab_src.encode(word_sentences_src[i], max_len)
            new_src = torch.tensor(new_src, dtype=torch.long).to(device)
            sentences_src.append(new_src)

            new_trg = vocab_trg.encode(word_sentences_trg[i], max_len)
            new_trg = torch.tensor(new_trg, dtype=torch.long).to(device)
            sentences_trg.append(new_trg)

        self.src = sentences_src
        self.trg = sentences_trg

        assert(len(self.src) == len(self.trg))

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]


class SubmissionDataset(Dataset):
    def __init__(self, filename, vocab, device='cuda'):
        word_sentences = Tokenizer(filename).sentences

        src = [vocab.encode(sentence) for sentence in word_sentences]
        src = [torch.tensor(sentence, dtype=torch.long).to(device) for sentence in src]
        self.src = src

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx]

class RawDataset(Dataset):
    def __init__(self, filename):
        self.src = Tokenizer(filename).sentences

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx]



def collate_fn_submission(batch):
    src_padded = pad_sequence(batch, batch_first=True, padding_value=pad_idx)
    return src_padded


class SubmissionDataLoader(DataLoader):
    def __init__(self, dataset):
        super().__init__(
            dataset,
            batch_size=128,
            shuffle=False,
            collate_fn=collate_fn_submission,
            num_workers=0,
        )


def collate_fn(batch):
    src_batch = [item[0] for item in batch]
    tgt_batch = [item[1] for item in batch]

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


class TrainDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=64, shuffle=True):
        super(TrainDataLoader, self).__init__(
            dataset, 
            batch_size=batch_size,  
            num_workers=0,
            shuffle=shuffle,
            collate_fn=collate_fn
        )


class TestDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=256, shuffle=False):
        super(TestDataLoader, self).__init__(
            dataset, 
            batch_size=batch_size,  
            num_workers=0,
            # pin_memory=True,
            shuffle=shuffle,
            collate_fn=collate_fn
        )