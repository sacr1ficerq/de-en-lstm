from lstm3 import train

from config import filenames, folders

from matplotlib import pyplot as plt
from dataset import Vocab, TranslationDataset

import psutil
import os

device = 'cuda'

config = {
    'model_name': 'LSTM_3',
    'feature': 'back-to-origins',
    'max_len': 42,
    'min_freq_src': 5,
    'min_freq_trg': 5,

    'embedding_dim': 192,
    'hidden_size': 320,
    'num_layers': 3,

    'num_epochs': 15,
    'weight_decay': 2e-5,
    'label_smoothing': 0.1,

    'dropout_emb': 0.2,

    'dropout_enc': 0.4,
    'dropout_dec': 0.4,

    'dropout_attention': 0.15,

    'learning_rate': 8e-4,
    'gamma': 0.2,
    'patience': 1,
    'threshold': 5e-4,
    'batch_size': 128,

    'use_tf': False,
    'tf_from_epoch': 0,
    'tf_start': 0.9,
    'tf_decrease': 0.02
}

vocab_src = Vocab(filenames['train_src'], min_freq=config['min_freq_src'])
vocab_trg = Vocab(filenames['train_trg'], min_freq=config['min_freq_trg'])

# config['weights'] = '../weights/saves/lstm-save-15.pt'

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
                                sort_lengths=False)

config['src_vocab_size'] = len(vocab_src)
config['trg_vocab_size'] = len(vocab_trg)

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

if __name__ == "__main__":
    # set current process to high priority
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows-specific

    train_losses, val_losses = train(config=config, 
                                     filenames=filenames, 
                                     folders=folders, 
                                     use_wandb=True, 
                                     device=device, 
                                     vocab_src=vocab_src,
                                     vocab_trg=vocab_trg,
                                     train_dataset=train_dataset,
                                     val_dataset=val_dataset)

    plot_losses(train_losses, val_losses)