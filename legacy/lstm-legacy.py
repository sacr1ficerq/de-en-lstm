from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

device = 'cuda'

bos_idx = 2
eos_idx = 3

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


def train(model, optimizer, num_epochs, train_loader, val_loader, criterion, trg_vocab_size):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for src_seq, trg_seq in tqdm(train_loader):
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


class LSTMModel(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embedding_dim, hidden_size):
        super(LSTMModel, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, embedding_dim)

        self.encoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)

        self.decoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, trg_vocab_size)

    def forward(self, src_seq, trg_seq):
        src_embedded = self.src_embedding(src_seq) # (batch_size, src_len, embedding_dim)
        trg_embedded = self.trg_embedding(trg_seq) # (batch_size, trg_len, embedding_dim)

        _, (hidden, cell) = self.encoder(src_embedded)  # hidden/cell: (1, batch_size, hidden_size)
 
        decoder_outputs, _ = self.decoder(trg_embedded, (hidden, cell))  # (batch_size, trg_len, hidden_size)

        logits = self.fc(decoder_outputs)  # (batch_size, trg_len, trg_vocab_size)
        return logits

    def inference(self, src_seq, max_len=50):
        src_seq = src_seq.to(device)
        self.eval()

        batch_size = src_seq.size(0)
        trg_seq = torch.tensor([[bos_idx]] * batch_size, dtype=torch.long).to(device)

        with torch.no_grad():
            src_embedded = self.src_embedding(src_seq)
            _, (hidden, cell) = self.encoder(src_embedded)

            for _ in range(max_len):
                trg_embedded = self.trg_embedding(trg_seq)  # (batch_size, current_len, embedding_dim)
                decoder_output, (hidden, cell) = self.decoder(trg_embedded, (hidden, cell))
                logits = self.fc(decoder_output[:, -1, :])  # (batch_size, trg_vocab_size)
                next_token = logits.argmax(dim=-1)  # greedy decoding
                trg_seq = torch.cat([trg_seq, next_token.unsqueeze(1)], dim=1)  # Append to sequence

                if (next_token == eos_idx).all():
                    break

        return trg_seq
    

class LSTM_Deep(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embedding_dim, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
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
        attention = F.softmax(energy, dim=-1)
        context = torch.bmm(attention, encoder_outputs)
        combined = torch.cat([decoder_outputs, context], dim=2)

        logits = self.fc(combined)
        return logits

    def inference(self, src_seq, max_len=50):
        src_seq = src_seq.to(device)
        self.eval()

        batch_size = src_seq.size(0)
        trg_seq = torch.tensor([[bos_idx]] * batch_size, dtype=torch.long).to(device)  # (batch_size, 1)

        with torch.no_grad():
            # encoder forward
            src_embedded = self.src_embedding(src_seq)
            encoder_outputs, (hidden, cell) = self.encoder(src_embedded)
            encoder_outputs = self.encoder_output_proj(encoder_outputs)
            encoder_outputs = encoder_outputs.contiguous()  # Ensure contiguous
            
            # project encoder hidden states for decoder
            hidden = self._project_hidden(hidden, self.encoder_hidden_proj)
            cell = self._project_hidden(cell, self.encoder_cell_proj)

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
                next_token = logits.argmax(dim=-1)
                
                trg_seq = torch.cat([trg_seq, next_token], dim=1)

                if (next_token == eos_idx).all():
                    break

        return trg_seq