from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm

import re
import random

import sentencepiece as spm

unk_idx, pad_idx, bos_idx, eos_idx, num_idx, sub_idx = 0, 1, 2, 3, 4, 5



specials = ['<UNK>', '<PAD>', '<BOS>', '<EOS>', '<NUM>', '<SUB>']


def tokenize_line(line):
    # print(line)
    return line.strip().split()

class BPE():
    def __init__(self, filename, vocab_size):
        spm.SentencePieceTrainer.train(
            input=filename,
            model_prefix='bpe_model',
            vocab_size=vocab_size,
            model_type='bpe',
            unk_id=unk_idx,
            pad_id=pad_idx,
            bos_id=bos_idx,
            eos_id=eos_idx,
            split_by_whitespace=False,
            # max_sentence_length=0,
            split_digits=False
        )

        sp = spm.SentencePieceProcessor()
        sp.load('bpe_model.model')
        self.tokenize = sp.encode_as_pieces
        self.detokenize = sp.decode_pieces

class Tokenizer():
    def __init__(self, filename, use_bpe=False, vocab_size=25000):
        if use_bpe:
            self.bpe = BPE(filename, vocab_size)
            self.tokenize_line = self.bpe.tokenize
        else:
            self.tokenize_line = tokenize_line
    def tokenize(self, filename):
        lines = []
        with open(filename, encoding='utf-8') as f:
            lines = f.readlines()
        sentences = [self.tokenize_line(' ' + line) for line in lines]
        return sentences

pattern_float = r"^ ?-?\+?\d+\.?\,?\d*\+?-?$"
pattern_has_digit = r"\d"

class Vocab():
    def __init__(self, filename, min_freq=2, max_freq_sub=8, sub_len=4, specials=specials, use_bpe=False, use_sub=False, vocab_size=25000):
        self.tokenizer = Tokenizer(filename, use_bpe=use_bpe, vocab_size=vocab_size)
        sentences = self.tokenizer.tokenize(filename)
        self.sub_len = sub_len
        self.max_freq_sub = max_freq_sub

        self.token_counter = Counter([token for tokens in sentences for token in tokens])
        self.vocab = {}
        def check_float(token):
            return re.match(pattern_float, token)

        self.check_float = check_float

        def check_sub(token):
            if not re.match(pattern_float, token) \
                and (re.match(pattern_has_digit, token) \
                or len(token) <= self.sub_len) \
                and self.token_counter[token] <= self.max_freq_sub: return True
            return False

        if use_sub:
            self.check_sub = check_sub
        else:
            self.check_sub = lambda token: False

        self.specials = set(specials)

        vocab_set = set(self.token_counter.keys())

        self.all_tokens = []
        for token in vocab_set:
            if self.check_float(token) \
            or self.check_sub(token) \
            or self.token_counter[token] < min_freq: continue
            self.all_tokens.append(token)
        self.all_tokens = specials + sorted(set(self.all_tokens) - self.specials)

        for i, token in enumerate(self.all_tokens):
            self.vocab[token] = i

    def plot_vocab(self):
        cnt = Counter(self.token_counter.values())
        xs = sorted(cnt.keys())
        ys = [cnt[k] for k in xs]
        plt.xlabel('frequency')
        plt.ylabel('amount of frequency')
        plt.plot(xs[:100], ys[:100])
        plt.show()
        
        # print((cnt[1]+2* cnt[2])/sum((np.array(list(cnt.values())))*(np.array(list(cnt.keys())))))

    def encode_token(self, token) -> int:
        if self.check_float(token):
            return num_idx
        if token not in self.vocab:
            if self.check_sub(token):
                return sub_idx
            return unk_idx
        return self.vocab[token]

    def decode_idx(self, idx) -> str:
        return self.all_tokens[idx]

    def encode_line(self, line, max_len=None):
        if len(line) == 0:
            return ['<BOS>', '<EOS>']
        tokens = self.tokenizer.tokenize_line(line)
        if tokens[0] != '<BOS>':
            tokens.insert(0, '<BOS>')
        if tokens[-1] != '<EOS>':
            tokens.append('<EOS>')
        
        if max_len:
            tokens = tokens[:max_len]
            # if tokens[-1] != '<EOS>':
            #     tokens[-1] = ('<EOS>')

        return [self.encode_token(token) for token in tokens]

    def encode(self, tokens, max_len=None):
        if len(tokens) == 0:
            return ['<BOS>', '<EOS>']
        if tokens[0] != '<BOS>':
            tokens.insert(0, '<BOS>')
        if tokens[-1] != '<EOS>':
            tokens.append('<EOS>')
        
        if max_len:
            tokens = tokens[:max_len]
            # if tokens[-1] != '<EOS>':
            #     tokens[-1] = ('<EOS>')

        return [self.encode_token(token) for token in tokens]

    def decode(self, idxs, ignore=set(['<UNK>', '<BOS>', '<EOS>', '<PAD>']), src=None, vocab_src=None):
        result = []

        has_num = False
        has_sub = False

        nums_idxs = []
        sub_idxs = []
        i = 0
        # if len(idxs) == 0: return []
        for idx in idxs:
            if i != 0 and idxs[i-1] == idxs[i]: continue 
            if idx == eos_idx:
                break
            if idx == num_idx:
                has_num = True
                nums_idxs.append(i)
            if idx == sub_idx:
                has_sub = True
                sub_idxs.append(i)
            result.append(self.decode_idx(idx))
            i += 1
        if has_num and src != None:
            nums_src = list(filter(self.check_float, src))
            # print(nums_idxs, len(result))
            for t, idx in enumerate(nums_idxs):
                if len(nums_src) <= t:
                    break
                result[idx] = nums_src[t]
        if has_sub and src != None:
            subs_src = list(filter(vocab_src.check_sub, src))
            for t, idx in enumerate(sub_idxs):
                if len(subs_src) <= t:
                    break
                result[idx] = subs_src[t]
        # if len(result) == 0: return result
        return list(filter(lambda token: not token in ignore, result))

    def __len__(self):
        return len(self.all_tokens)


class TranslationDataset(Dataset):
    def __init__(self, vocab_src, vocab_trg, filename_src, filename_trg, max_len=100, device='cuda', sort_lengths=False):
        self.max_len = max_len
        token_sentences_src = vocab_src.tokenizer.tokenize(filename_src)
        token_sentences_trg = vocab_trg.tokenizer.tokenize(filename_trg)

        lengths_src = [len(sentence) for sentence in token_sentences_src]
        lengths_trg = [len(sentence) for sentence in token_sentences_trg]
        
        self.cnt_src = Counter(lengths_src)
        self.cnt_trg = Counter(lengths_trg)

        sentences_src = []
        sentences_trg = []

        self.ids = []

        if sort_lengths:
            n = len(token_sentences_src)
            self.ids = sorted(range(n), key=lambda i: len(token_sentences_src[i]), reverse=True) # descending by src length
            token_sentences_src = [token_sentences_src[i] for i in self.ids]
            token_sentences_trg = [token_sentences_trg[i] for i in self.ids]


        for i in tqdm(range(len(token_sentences_src))):
            new_src = vocab_src.encode(token_sentences_src[i], max_len)
            new_src = torch.tensor(new_src, dtype=torch.long).to(device)
            sentences_src.append(new_src)

            new_trg = vocab_trg.encode(token_sentences_trg[i], max_len)
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
        token_sentences = Tokenizer(filename).sentences

        src = [vocab.encode(sentence) for sentence in token_sentences]
        src = [torch.tensor(sentence, dtype=torch.long).to(device) for sentence in src]
        self.src = src

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx]

class RawDataset(Dataset):
    def __init__(self, filename, n=int(1e3)):
        print(n)
        self.src = Tokenizer(filename).tokenize(filename)[:n+1]

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
    def __getitem__(self, idx):
        for i, batch in enumerate(self):
            if i == idx:
                return batch
        return None

def collate_bucket(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True) # descending lengths
    src_batch, trg_batch = zip(*batch)

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, trg_padded

def bucket_iterator(dataset, batch_size=128, shuffle=True):
    n = len(dataset)
    sorted_id = sorted(range(n), key=lambda i: len(dataset[i][0]), reverse=True) # descending by src length
    sorted_dataset = [dataset[i] for i in sorted_id]
    
    buckets = [sorted_dataset[i:i+batch_size] for i in range(0, n, batch_size)]

    if shuffle:
        random.shuffle(buckets)
    
    for bucket in buckets:
        yield collate_bucket(bucket)