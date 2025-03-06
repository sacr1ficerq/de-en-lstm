from lstm2 import train

from config import filenames, folders

from matplotlib import pyplot as plt

import psutil
import os

device = 'cuda'

config = {
    'model_name': 'LSTM_2',
    'feature': 'new-vocab',
    'max_len': 48,
    'min_freq_src': 5,
    'min_freq_trg': 4,

    'src_vocab_size': 24991,
    'trg_vocab_size': 21555,

    'embedding_dim': 128,
    'hidden_size': 256,
    'num_layers': 3,

    'num_epochs': 18,
    'weight_decay': 1e-5,
    'label_smoothing': 0.1,
    'dropout': 0.25,

    'learning_rate': 1e-3,
    'gamma': 0.2,
    'patience': 0,
    'threshold': 0.001
}

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

# set current process to high priority
p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows-specific

train_losses, val_losses = train(config=config, filenames=filenames, folders=folders, use_wandb=True, device=device)

plot_losses(train_losses, val_losses)