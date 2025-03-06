from collections import Counter
from torch.utils.data import Dataset
import torch

def tokenize(line):
    return ['<BOS>'] + line.strip().split() + ['<EOS>']

class TranslationDataset(Dataset):
    def __init__(self, src_filename, trg_filename, min_freq=2):
        train_src_lines = []
        train_trg_lines = []

        with open(src_filename, encoding='utf-8') as src_f, open(trg_filename, encoding='utf-8') as trg_f:
            train_src_lines = src_f.readlines()
            train_trg_lines = trg_f.readlines()

        train_src = [tokenize(line) for line in train_src_lines]
        train_trg = [tokenize(line) for line in train_trg_lines]

        counter_src = Counter([word for words in train_src for word in words])
        counter_trg = Counter([word for words in train_trg for word in words])


        self.vocab_src = {}
        self.vocab_trg = {}

        self.specials = ['<UNK>', '<PAD>', '<BOS>', '<EOS>']

        all_words_src = set(counter_src.keys())
        all_words_trg = set(counter_trg.keys())

        self.vocab_src_words = self.specials + sorted(set([word if counter_src[word] >= min_freq else '<UNK>' for word in all_words_src]) - set(self.specials))
        self.vocab_trg_words = self.specials + sorted(set([word if counter_trg[word] >= min_freq else '<UNK>' for word in all_words_trg]) - set(self.specials))

        self.unk_idx, self.pad_idx, self.bos_idx, self.eos_idx = 0, 1, 2, 3

        for i, word in enumerate(self.vocab_src_words):
            self.vocab_src[word] = i

        for i, word in enumerate(self.vocab_trg_words):
            self.vocab_trg[word] = i

        self.src = train_src
        self.trg = train_trg

    def encode_word_src(self, word):
        return self.vocab_src[word] if word in self.vocab_src else self.unk_idx
    def encode_word_trg(self, word):
        return self.vocab_trg[word] if word in self.vocab_trg else self.unk_idx

    def decode_idx_src(self, idx):
        return self.vocab_src_words[idx] if idx < len(self.vocab_src_words) else self.unk_idx
    def decode_idx_trg(self, idx):
        return self.vocab_trg_words[idx] if idx < len(self.vocab_trg_words) else self.unk_idx


    def encode_src(self, words):
        return [self.bos_idx] + [self.encode_word_src(word) for word in words] + [self.eos_idx]
    def encode_trg(self, words):
        return [self.bos_idx] + [self.encode_word_trg(word) for word in words] + [self.eos_idx]

    def decode_src(self, idxs):
        return [self.decode_idx_src(idx) for idx in idxs]
    def decode_trg(self, idxs):
        return [self.decode_idx_trg(idx) for idx in idxs]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]



class PairsDataset(Dataset):
    def __init__(self, source_sequences, target_sequences):
        self.source = source_sequences
        self.target = target_sequences

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]
        return src, tgt

# import matplotlib.pyplot as plt
# cnt = Counter(counter_src.values())
# xs = sorted(cnt.keys())
# ys = [cnt[k] for k in xs]
# plt.xlabel('frequency')
# plt.ylabel('amount of frequency')
# plt.plot(xs[:100], ys[:100])

class SubmissionDataset(Dataset):
    def __init__(self, val_filename, train_dataset):
        val_src_lines = []

        with open(val_filename) as src_f:
            val_src_lines = src_f.readlines()

        val_src = [tokenize(line) for line in val_src_lines]
        val_src = [train_dataset.encode_src(line) for line in val_src]
        val_src = [torch.tensor(line, dtype=torch.long) for line in val_src]
        self.source = val_src

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = self.source[idx]
        return src

