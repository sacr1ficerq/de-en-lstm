from submission import get_bleu
from dataset import TranslationDataset, Vocab, TrainDataLoader, TestDataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

import numpy as np

from random import random

import wandb

unk_idx, pad_idx, bos_idx, eos_idx = 0, 1, 2, 3



def train_epoch(model, optimizer, train_loader, val_loader, criterion, vocab_trg, scheduler, teacher_forcing=1):
    model.train()
    total_train_loss = 0

    for src_seq, trg_seq in tqdm(train_loader):
        trg_input = trg_seq[:, :-1]
        trg_output = trg_seq[:, 1:]

        logits = model(src_seq, trg_input, teacher_forcing=teacher_forcing)
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

            logits = model(src_seq, trg_input, teacher_forcing=0)
            logits = logits.view(-1, len(vocab_trg))
            trg_output = trg_output.reshape(-1)

            loss = criterion(logits, trg_output)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_train_loss = total_train_loss / len(train_loader)

    if scheduler:
        scheduler.step(avg_val_loss)

    return avg_train_loss, avg_val_loss

def train(config, filenames, folders, device='cuda', use_wandb=False, vocab_src=None, vocab_trg=None, train_dataset=None, val_dataset=None):
    if vocab_src == None: vocab_src = Vocab(filenames['train_src'], min_freq=config['min_freq_src'])
    if vocab_src == None: vocab_trg = Vocab(filenames['train_trg'], min_freq=config['min_freq_trg'])

    if train_dataset==None: train_dataset = TranslationDataset(vocab_src, 
                                    vocab_trg, 
                                    filenames['train_src'], 
                                    filenames['train_trg'], 
                                    max_len=config['max_len'], 
                                    device=device)

    if val_dataset==None: val_dataset = TranslationDataset(vocab_src, 
                                    vocab_trg, 
                                    filenames['test_src'], 
                                    filenames['test_trg'], 
                                    max_len=72, 
                                    device=device, 
                                    sort_lengths=False)

    unk_idx, pad_idx, bos_idx, eos_idx = 0, 1, 2, 3

    train_loader = TrainDataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = TestDataLoader(val_dataset, batch_size=256, shuffle=False)

    config['src_vocab_size'] = len(vocab_src)
    config['trg_vocab_size'] = len(vocab_trg)

    model = LSTM_3(config=config).to(device)

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

    teacher_forcing = 1.0

    for epoch in range(1, config['num_epochs']+1):
        train_loss, val_loss = train_epoch(model, 
                                           optimizer, 
                                           train_loader, 
                                           val_loader, 
                                           criterion, 
                                           vocab_trg, 
                                           scheduler,
                                           teacher_forcing=teacher_forcing)
        
        val_losses.append(train_loss)
        train_losses.append(val_loss)

        if use_wandb: 
            wandb.log({"train_loss": train_loss,
                       "val_loss": val_loss, 
                       "epoch": epoch,})

            if scheduler:
                wandb.log({"lr": scheduler.get_last_lr()[0]})

        if epoch >= 4:
            teacher_forcing *= 0.8
            checkpoint_path = f'lstm-save-{epoch}.pt'
            model.save(checkpoint_path, folders['weights'])
            # if use_wandb: wandb.save(checkpoint_path)
            if epoch%4 == 0:
                try:
                    bleu_score = get_bleu(model, val_loader, vocab_trg, filenames)
                    print(f"BLEU4: {bleu_score}")
                    if use_wandb: wandb.log({"BLEU4": bleu_score})
                except:
                    print('wrong bleu function')

        print(f"Epoch [{epoch}/{config['num_epochs']}]\tTrain Loss: {train_loss:.4f}\tVal Loss: {val_loss:.4f}")


    model.train_loss = np.array(train_losses)
    model.val_loss = np.array(val_losses)

    model.save(weights_filename, folders['weights'])
    wandb.finish()

    return train_losses, val_losses

unk_idx, pad_idx, bos_idx, eos_idx, num_idx = 0, 1, 2, 3, 4


class LSTM_3(nn.Module):
    def __init__(self, 
                 src_vocab_size=None, 
                 trg_vocab_size=None, 
                 embedding_dim=None, 
                 hidden_size=None, 
                 num_layers=None,
                 enc_dropout=None,
                 dec_dropout=None, 
                 emb_dropout=None,
                 attention_dropout=None, 
                 config=None,
                 pad_idx=pad_idx,
                 weights_filename=None):
        super().__init__()

        self.train_loss = []
        self.val_loss = []

        if config:
            if not src_vocab_size:
                src_vocab_size = config['src_vocab_size']
            if not trg_vocab_size:
                trg_vocab_size = config['trg_vocab_size']
            if not embedding_dim:
                embedding_dim = config['embedding_dim']
            if not hidden_size:
                hidden_size = config['hidden_size']
            if not num_layers:
                num_layers = config.get('num_layers', 2)
            if not enc_dropout:
                enc_dropout = config.get('enc_dropout', 0.1)
            if not dec_dropout:
                dec_dropout = config.get('dec_dropout', 0.1)
            if not emb_dropout:
                emb_dropout = config.get('emb_dropout', 0.1)
            if not attention_dropout:
                attention_dropout = config.get('attention_dropout', 0.1)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim, padding_idx=pad_idx)
        self.trg_embedding = nn.Embedding(trg_vocab_size, embedding_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.enc_dropout = nn.Dropout(enc_dropout)
        self.dec_dropout = nn.Dropout(dec_dropout)

        self.attention_dropout = nn.Dropout(attention_dropout)

        self.encoder = nn.LSTM(embedding_dim, 
                               hidden_size, 
                               batch_first=True, 
                               bidirectional=True, 
                               num_layers=num_layers, 
                               dropout=enc_dropout)
        encoder_output_size = hidden_size * 2
        self.encoder_output_proj = nn.Linear(encoder_output_size, hidden_size)

        self.decoder = nn.LSTM(embedding_dim, 
                               hidden_size, 
                               batch_first=True,
                               num_layers=num_layers, 
                               dropout=dec_dropout)
        
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
        
        if weights_filename:
            self.load(weights_filename)

    def _project_hidden(self, state, proj_layers):
        batch_size = state.size(1)
        state = state.view(self.num_layers, 2, batch_size, self.hidden_size)
        state = torch.cat([state[:, 0], state[:, 1]], dim=2)  # (num_layers, batch, 2*hidden_size)
        return torch.stack([proj_layers[i](state[i]) for i in range(self.num_layers)], dim=0)

    def forward(self, src_seq, trg_seq, device='cuda', teacher_forcing=1.0):
        src_embedded = self.emb_dropout(self.src_embedding(src_seq)) # (batch_size, src_len, embedding_dim)

        encoder_outputs, (hidden, cell) = self.encoder(src_embedded)  # hidden/cell: (1, batch_size, hidden_size)
        encoder_outputs = self.enc_dropout(self.encoder_output_proj(encoder_outputs))

        hidden = self._project_hidden(hidden, self.encoder_hidden_proj)
        cell = self._project_hidden(cell, self.encoder_cell_proj)

        batch_size = src_seq.size(0)
        trg_len = trg_seq.size(1)
        trg_vocab_size = self.fc.out_features

        # tensor to store decoder outputs
        logits = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)
        decoder_input = trg_seq[:, 0].unsqueeze(1)  # [<sos>] (batch_size, 1)

        # autoregressive decoding with teacher forcing
        for t in range(1, trg_len):  # skip <sos>
            trg_embedded = self.emb_dropout(self.trg_embedding(decoder_input))  # (batch_size, 1, emb_dim)
            decoder_output, (hidden, cell) = self.decoder(trg_embedded, (hidden, cell))

            decoder_output = self.dec_dropout(decoder_output)

            energy = torch.bmm(decoder_output, encoder_outputs.transpose(1, 2))  # (batch_size, 1, src_len)

            mask = (src_seq != pad_idx).unsqueeze(1)  # (batch_size, 1, src_len)
            energy = energy.masked_fill(mask == 0, -1e10)
            
            attention = F.softmax(energy, dim=-1)
            attention = self.attention_dropout(attention)
            context = torch.bmm(attention, encoder_outputs)  # (batch_size, 1, hidden_dim)

            combined = torch.cat([decoder_output, context], dim=2)  # (batch_size, 1, hidden_dim * 2)
            step_logits = self.fc(combined)  # (batch_size, 1, trg_vocab_size)

            logits[:, t] = step_logits.squeeze(1)

            use_teacher_forcing = random() < teacher_forcing
            top1 = step_logits.argmax(-1)  # (batch_size, 1)
            decoder_input = trg_seq[:, t].unsqueeze(1) if use_teacher_forcing else top1

        return logits  # (batch_size, trg_len, trg_vocab_size)

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

    def load(self, filename):
        self.load_state_dict(torch.load(filename, weights_only=True))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)