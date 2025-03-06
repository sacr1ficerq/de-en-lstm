from dataset import TranslationDataset, PairsDataset, SubmissionDataset
from lstm import train, LSTM_Deep

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

import zipfile
import os

from sklearn.model_selection import train_test_split


device = 'cuda'

zip_path = 'bhw2-data.zip'

extract_to = './'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"Files extracted to {extract_to}")

data_folder = extract_to + './data/'
print("Extracted files:", os.listdir(data_folder))

train_src_filename = data_folder + 'train.de-en.de'
train_trg_filename = data_folder + 'train.de-en.en'


train_dataset = TranslationDataset(train_src_filename, train_trg_filename, min_freq=2)

X = [torch.tensor(train_dataset.encode_src(seq), dtype=torch.long) for seq in train_dataset.src]
y = [torch.tensor(train_dataset.encode_trg(seq), dtype=torch.long) for seq in train_dataset.trg]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=12)
unk_idx, pad_idx, bos_idx, eos_idx = 0, 1, 2, 3

def collate_fn(batch):
    src_batch = [item[0] for item in batch]
    tgt_batch = [item[1] for item in batch]
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded

train_pairs = PairsDataset(x_train, y_train)
train_loader = DataLoader(
    train_pairs,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=0,
)

test_pairs = PairsDataset(x_test, y_test)
test_loader = DataLoader(
    test_pairs,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=0,
)

src_vocab_size = len(train_dataset.vocab_src)
trg_vocab_size = len(train_dataset.vocab_trg)
embedding_dim = 64
hidden_size = 128
learning_rate = 0.001
num_epochs = 1

model = LSTM_Deep(
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)  # Ignore padding index in loss computation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# model.load_state_dict(torch.load('lstm-full-vocab-7epoch.pt', weights_only=True))
train(model, optimizer, num_epochs, train_loader, test_loader, criterion, len(train_dataset.vocab_trg))

val_filename = data_folder + 'test1.de-en.de'

pad_idx = 1
def collate_fn_submission(batch):
    src_padded = pad_sequence(batch, batch_first=True, padding_value=pad_idx)
    return src_padded

submission_dataset = SubmissionDataset(val_filename, train_dataset)
submission_loader = DataLoader(
    submission_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=collate_fn_submission,
    pin_memory=True,
    num_workers=0,
)

specials = set(['<UNK>', '<PAD>', '<BOS>', '<EOS>'])
submission_filename = 'submission.en'
def make_submission(model, submission_loader, train_dataset, submission_filename=submission_filename):
    with open(submission_filename, 'w', encoding='utf-8') as f_pred:
        with torch.no_grad():
            for a in tqdm(submission_loader):
                a = a.to(device)
                predictions = model.inference(a) # batch
                for i in range(len(a)):
                    pred_text = train_dataset.decode_trg(predictions[i][1:])
                    pred_text = filter(lambda word: not word in specials, pred_text)
                    f_pred.write(" ".join(pred_text) + '\n')

make_submission(model, submission_loader, train_dataset)