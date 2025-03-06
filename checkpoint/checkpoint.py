import zipfile
# Path to the zip file
zip_path = 'bhw2-data.zip'

# Directory to extract the contents to
extract_to = './'

# Unzip the file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"Files extracted to {extract_to}")


import os

# List files in the extracted directory
data_folder = extract_to + './data/'
print("Extracted files:", os.listdir(data_folder))

train_src_filename = data_folder + 'train.de-en.de'
train_trg_filename = data_folder + 'train.de-en.en'

train_src_lines = []
train_trg_lines = []

with open(train_src_filename, encoding='utf-8') as src_f, open(train_trg_filename, encoding='utf-8') as trg_f:
    train_src_lines = src_f.readlines()
    train_trg_lines = trg_f.readlines()


def tokenize(line):
    return ['<BOS>'] + line.strip().split() + ['<EOS>']

train_src = [tokenize(line) for line in train_src_lines]
train_trg = [tokenize(line) for line in train_trg_lines]

from collections import Counter
counter_src = Counter([word for words in train_src for word in words])
counter_trg = Counter([word for words in train_trg for word in words])

import matplotlib.pyplot as plt
cnt = Counter(counter_src.values())
xs = sorted(cnt.keys())
ys = [cnt[k] for k in xs]

plt.xlabel('frequency')
plt.ylabel('amount of frequency')
plt.plot(xs[:100], ys[:100])

vocab_src = {}
vocab_trg = {}

vocab_src_words = ['<UNK>', '<PAD>', '<BOS>', '<EOS>'] + sorted(set([word if counter_src[word] > 2 else '-' for word in set(counter_src.keys())]) - set(['<UNK>', '<PAD>', '<BOS>', '<EOS>']))
vocab_trg_words = ['<UNK>', '<PAD>', '<BOS>', '<EOS>'] + sorted(set([word if counter_trg[word] > 2 else '-' for word in set(counter_trg.keys())]) - set(['<UNK>', '<PAD>', '<BOS>', '<EOS>']))

unk_idx, pad_idx, bos_idx, eos_idx = 0, 1, 2, 3

for i, word in enumerate(vocab_src_words):
    vocab_src[word] = i

for i, word in enumerate(vocab_trg_words):
    vocab_trg[word] = i


def encode_word_src(word):
    return vocab_src[word] if word in vocab_src else unk_idx
def encode_word_trg(word):
    return vocab_trg[word] if word in vocab_trg else unk_idx

def decode_idx_src(idx):
    return vocab_src_words[idx] if idx < len(vocab_src_words) else unk_idx
def decode_idx_trg(idx):
    return vocab_trg_words[idx] if idx < len(vocab_trg_words) else unk_idx


def encode_src(words):
    return [bos_idx] + [encode_word_src(word) for word in words] + [eos_idx]
def encode_trg(words):
    return [bos_idx] + [encode_word_trg(word) for word in words] + [eos_idx]

def decode_src(idxs):
    return [decode_idx_src(idx) for idx in idxs]
def decode_trg(idxs):
    return [decode_idx_trg(idx) for idx in idxs]


import torch # 2.5.1+cu124

from sklearn.model_selection import train_test_split

X = [torch.tensor(encode_src(seq), dtype=torch.long) for seq in train_src]
y = [torch.tensor(encode_trg(seq), dtype=torch.long) for seq in train_trg]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)


from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Dataset
class TranslationDataset(Dataset):
    def __init__(self, source_sequences, target_sequences):
        self.source = source_sequences
        self.target = target_sequences

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]
        return src, tgt

# Collate function
def collate_fn(batch):
    src_batch = [item[0] for item in batch]
    tgt_batch = [item[1] for item in batch]
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=vocab_src['<PAD>'])
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=vocab_src['<PAD>'])
    return src_padded, tgt_padded

# DataLoader
train_dataset = TranslationDataset(x_train, y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=2,
)

test_dataset = TranslationDataset(x_test, y_test)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=2,
)

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()

import torch.optim as optim
from tqdm import tqdm

device = 'cuda'

import torch.nn as nn
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)  # Ignore padding index in loss computation

src_vocab_size = len(vocab_src)
trg_vocab_size = len(vocab_trg)

def train(model, optimizer, num_epochs, train_loader, val_loader):
    train_losses = []
    val_losses = []

    # training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch_idx, (src_seq, trg_seq) in enumerate(tqdm(train_loader)):
            src_seq = src_seq.to(device)
            trg_seq = trg_seq.to(device)

            trg_input = trg_seq[:, :-1]
            trg_output = trg_seq[:, 1:]

            logits = model(src_seq, trg_input)
            logits = logits.view(-1, trg_vocab_size)
            trg_output = trg_output.reshape(-1)

            loss = criterion(logits, trg_output)
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for src_seq, trg_seq in val_loader:
                src_seq = src_seq.to(device)
                trg_seq = trg_seq.to(device)

                trg_input = trg_seq[:, :-1]
                trg_output = trg_seq[:, 1:]

                logits = model(src_seq, trg_input)
                logits = logits.view(-1, trg_vocab_size)
                trg_output = trg_output.reshape(-1)

                loss = criterion(logits, trg_output)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    plot_losses(train_losses, val_losses)




import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embedding_dim, hidden_size):
        super(LSTMModel, self).__init__()
        self.emb_layer_norm = nn.LayerNorm(embedding_dim)  # Add layer norm

        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, embedding_dim)

        self.hidden_proj = nn.Linear(hidden_size, hidden_size)
        self.cell_proj = nn.Linear(hidden_size, hidden_size)

        # Encoder LSTM
        self.encoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

        # Decoder LSTM
        self.decoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

        # Final prediction layer
        self.fc = nn.Linear(hidden_size, trg_vocab_size)

    def forward(self, src_seq, trg_seq):
        # Encoder
        # src_embedded = self.src_embedding(src_seq)  # (batch_size, src_len, embedding_dim)
        src_embedded = self.emb_layer_norm(self.src_embedding(src_seq))
        trg_embedded = self.emb_layer_norm(self.trg_embedding(trg_seq))

        _, (hidden, cell) = self.encoder(src_embedded)  # hidden/cell: (1, batch_size, hidden_size)
        hidden = torch.tanh(self.hidden_proj(hidden))  # Project hidden state
        cell = torch.tanh(self.cell_proj(cell))         # Project cell state

        # Decoder
        # trg_embedded = self.trg_embedding(trg_seq)  # (batch_size, trg_len, embedding_dim)
        decoder_outputs, _ = self.decoder(trg_embedded, (hidden, cell))  # (batch_size, trg_len, hidden_size)

        # Final predictions
        logits = self.fc(decoder_outputs)  # (batch_size, trg_len, trg_vocab_size)
        return logits
  
    def inference(self, src_seq, max_len=50):
        src_seq = src_seq.to(device)  # Move to the appropriate device (e.g., GPU)
        model.eval()  # Set the model to evaluation mode
        trg_seq = [[bos_idx] for batch_id in range(src_seq.size()[0])]

        with torch.no_grad():  # Disable gradient computation
            # src_seq = src_seq.unsqueeze(0)  # Add batch dimension: (1, src_seq_len)
            for _ in range(max_len):
                # Prepare the target sequence so far
                trg_seq_tensor = torch.tensor(trg_seq, dtype=torch.long)  # (1, current_seq_len)
                trg_seq_tensor = trg_seq_tensor.to(device)
                # Predict the next token
                logits = model(src_seq, trg_seq_tensor)  # (1, current_seq_len, trg_vocab_size)
                next_token_logits = logits[:, -1, :]  # Get logits for the last token: (1, trg_vocab_size)
                next_token = next_token_logits.argmax(dim=-1)  # Greedy decoding

                # Stop if <eos> token is generated
                # if next_token == eos_idx:
                #     break
                # Append the generated token to the target sequence
                for batch_id in range(src_seq.size()[0]):
                  trg_seq[batch_id].append(next_token[batch_id])

        return trg_seq
    


# Hyperparameters
src_vocab_size = len(vocab_src)
trg_vocab_size = len(vocab_trg)
embedding_dim = 64
hidden_size = 128
learning_rate = 0.01
num_epochs = 5

# Initialize model, loss, and optimizer
model = LSTMModel(
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)  # Ignore padding index in loss computation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


train(model, optimizer, 3, train_loader, test_loader)