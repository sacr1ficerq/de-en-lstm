from submission import get_bleu
from dataset import TranslationDataset, Vocab, TrainDataLoader, TestDataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

import numpy as np

import wandb

unk_idx, pad_idx, bos_idx, eos_idx = 0, 1, 2, 3



def train_epoch(model, optimizer, train_loader, val_loader, criterion, vocab_trg, scheduler):
    model.train()
    total_train_loss = 0

    for src_seq, trg_seq in tqdm(train_loader):
        trg_input = trg_seq[:, :-1]
        trg_output = trg_seq[:, 1:]

        logits = model(src_seq, trg_input)
        logits = logits.view(-1, len(vocab_trg))
        trg_output = trg_output.reshape(-1)

        loss = criterion(logits, trg_output)
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for src_seq, trg_seq in tqdm(val_loader):
            trg_input = trg_seq[:, :-1]
            trg_output = trg_seq[:, 1:]

            logits = model(src_seq, trg_input)
            logits = logits.view(-1, len(vocab_trg))
            trg_output = trg_output.reshape(-1)

            loss = criterion(logits, trg_output)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_train_loss = total_train_loss / len(train_loader)

    if scheduler:
        scheduler.step(avg_val_loss)

    return avg_train_loss, avg_val_loss

def train(config, filenames, folders, device='cuda', use_wandb=False):
    vocab_src = Vocab(filenames['train_src'], min_freq=config['min_freq_src'])
    vocab_trg = Vocab(filenames['train_trg'], min_freq=config['min_freq_trg'])

    train_dataset = TranslationDataset(vocab_src, 
                                    vocab_trg, 
                                    filenames['train_src'], 
                                    filenames['train_trg'], 
                                    max_len=config['max_len'], 
                                    device=device)

    val_dataset = TranslationDataset(vocab_src, 
                                    vocab_trg, 
                                    filenames['test_src'], 
                                    filenames['test_trg'], 
                                    max_len=72, 
                                    device=device, 
                                    sort_lengths=True)

    unk_idx, pad_idx, bos_idx, eos_idx = 0, 1, 2, 3

    train_loader = TrainDataLoader(train_dataset, shuffle=True)
    val_loader = TestDataLoader(val_dataset)

    src_vocab_size = len(vocab_src)
    trg_vocab_size = len(vocab_trg)

    model = LSTM_2(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        dropout=config['dropout']
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, 
                                    label_smoothing=config['label_smoothing'])

    optimizer = optim.AdamW(model.parameters(), 
                            lr=config['learning_rate'], 
                            weight_decay=config['weight_decay'])

    scheduler = ReduceLROnPlateau(optimizer, 
                                patience=config['patience'], 
                                factor=config['gamma'], 
                                threshold=config['threshold'])

    print(model)

    params_n = model.count_parameters()
    weights_filename = folders['weights'] + f"{config['model_name']}-{config['feature']}-{params_n // 1e+6}m-{config['num_epochs']}epoch.pt"

    if use_wandb: wandb.init(
        project="bhw2",
        config=config
    )
    
    # if use_wandb: wandb.watch(model, log="parameters", log_freq=10)
    train_losses = []
    val_losses = []

    for epoch in range(config['num_epochs']):
        train_loss, val_loss = train_epoch(model, optimizer, train_loader, val_loader, criterion, vocab_trg, scheduler)
        
        val_losses.append(train_loss)
        train_losses.append(val_loss)

        if use_wandb: 
            wandb.log({"train_loss": train_loss,
                       "val_loss": val_loss, 
                       "epoch": epoch,})

            if scheduler:
                wandb.log({"lr": scheduler.get_last_lr()[0]})

        if epoch > 5:
            checkpoint_path = f'lstm-save-{epoch}.pt'
            model.save(checkpoint_path, folders['weights'])
            # if use_wandb: wandb.save(checkpoint_path)
            if epoch%3 == 0:
                bleu_score = get_bleu(model, val_loader, vocab_trg)
                print(f"BLEU4: {bleu_score}")
                if use_wandb: wandb.log({"BLEU4": bleu_score})

        print(f"Epoch [{epoch+1}/{config['num_epochs']}]\tTrain Loss: {train_loss:.4f}\tVal Loss: {val_loss:.4f}")


    model.train_loss = np.array(train_losses)
    model.val_loss = np.array(val_losses)

    model.save(weights_filename, folders['weights'])
    wandb.finish()

    return train_losses, val_losses


class LSTM_2(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embedding_dim, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()

        self.train_loss = []
        self.val_loss = []

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, embedding_dim)

        self.encoder = nn.LSTM(embedding_dim, 
                               hidden_size, 
                               batch_first=True, 
                               bidirectional=True, 
                               num_layers=num_layers, 
                               dropout=dropout)
        encoder_output_size = hidden_size * 2
        self.encoder_output_proj = nn.Linear(encoder_output_size, hidden_size)

        self.decoder = nn.LSTM(embedding_dim, 
                               hidden_size, 
                               batch_first=True,
                               num_layers=num_layers, 
                               dropout=dropout)
        
        self.encoder_hidden_proj = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size) for _ in range(num_layers)
        ])
        self.encoder_cell_proj = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_size * 2, trg_vocab_size)

        # xavier
        for name, param in self.named_parameters():
            if "weight" in name and "embedding" not in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)


    def _project_hidden(self, state, proj_layers):
        batch_size = state.size(1)
        state = state.view(self.num_layers, 2, batch_size, self.hidden_size)
        state = torch.cat([state[:, 0], state[:, 1]], dim=2)  # (num_layers, batch, 2*hidden_size)
        return torch.stack([proj_layers[i](state[i]) for i in range(self.num_layers)], dim=0)

    def forward(self, src_seq, trg_seq):
        src_embedded = self.src_embedding(src_seq) # (batch_size, src_len, embedding_dim)

        encoder_outputs, (hidden, cell) = self.encoder(src_embedded)  # hidden/cell: (1, batch_size, hidden_size)
        encoder_outputs = self.encoder_output_proj(encoder_outputs)

        hidden = self._project_hidden(hidden, self.encoder_hidden_proj)
        cell = self._project_hidden(cell, self.encoder_cell_proj)

        trg_embedded = self.trg_embedding(trg_seq) # (batch_size, trg_len, embedding_dim)
        decoder_outputs, _ = self.decoder(trg_embedded, (hidden, cell))  # (batch_size, trg_len, hidden_size)

        # attention
        energy = torch.bmm(decoder_outputs, encoder_outputs.transpose(1, 2))

        mask = (src_seq != pad_idx).unsqueeze(1)  # (batch_size, 1, src_len)
        energy = energy.masked_fill(mask == 0, -1e10)

        attention = F.softmax(energy, dim=-1)
        context = torch.bmm(attention, encoder_outputs)
        combined = torch.cat([decoder_outputs, context], dim=2)

        logits = self.fc(combined)
        return logits

    def inference(self, src_seq, max_len=50, device='cuda'):
        self.eval()

        batch_size = src_seq.size(0)
        trg_seq = torch.tensor([[bos_idx]] * batch_size, dtype=torch.long).to(device)  # (batch_size, 1)

        with torch.no_grad():
            # encoder forward
            src_embedded = self.src_embedding(src_seq)
            encoder_outputs, (hidden, cell) = self.encoder(src_embedded)
            encoder_outputs = self.encoder_output_proj(encoder_outputs)
            encoder_outputs = encoder_outputs.contiguous()  # ensure contiguous
            
            # project encoder hidden states for decoder
            hidden = self._project_hidden(hidden, self.encoder_hidden_proj)
            cell = self._project_hidden(cell, self.encoder_cell_proj)
            if max_len == None:
                max_len = src_seq.size(1) + 5
            for _ in range(max_len):
                # get last token (batch_size, 1)
                current_trg = trg_seq[:, -1].unsqueeze(1)
                trg_embedded = self.trg_embedding(current_trg)  # (batch_size, 1, emb_dim)
                
                decoder_output, (hidden, cell) = self.decoder(trg_embedded, (hidden, cell))
                
                energy = torch.bmm(decoder_output, encoder_outputs.transpose(1, 2).contiguous())
                attention = F.softmax(energy, dim=-1)
                context = torch.bmm(attention, encoder_outputs)
                combined = torch.cat([decoder_output, context], dim=2)
                
                logits = self.fc(combined)  # (batch_size, 1, trg_vocab_size)
                logits[:, :, unk_idx] = -1e10
                next_token = logits.argmax(dim=-1)
                
                trg_seq = torch.cat([trg_seq, next_token], dim=1)

                if (next_token == eos_idx).all():
                    break

        return trg_seq

    def save(self, filename, folder):
        np.save(folder + 'train.npy', self.train_loss)
        np.save(folder + 'val.npy', self.val_loss)
        torch.save(self.state_dict(), folder + filename)

    def load(self, filename, folder):
        self.train_loss = np.load(folder + 'train.npy')
        self.val_loss = np.load(folder + 'val.npy')
        self.load_state_dict(torch.load(folder + filename, weights_only=True))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)