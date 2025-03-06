from dataset import TranslationDataset, Vocab
from matplotlib import pyplot as plt

import psutil
import os

import torch.nn as nn
from lstm2 import LSTM_2, train
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import TrainDataLoader, TestDataLoader

import wandb

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

def build(config, filenames, device='cuda', use_wandb=False):
    vocab_src = Vocab(filenames['train_src'], min_freq=config['min_freq_src'])
    vocab_trg = Vocab(filenames['train_trg'], min_freq=config['min_freq_trg'])

    train_dataset = TranslationDataset(vocab_src, 
                                    vocab_trg, 
                                    filenames['train_src'], 
                                    filenames['train_trg'], 
                                    max_len=config['max_len'], 
                                    device=device)

    test_dataset = TranslationDataset(vocab_src, 
                                    vocab_trg, 
                                    filenames['test_src'], 
                                    filenames['test_trg'], 
                                    max_len=72, 
                                    device=device, 
                                    sort_lengths=True)

    unk_idx, pad_idx, bos_idx, eos_idx = 0, 1, 2, 3

    train_loader = TrainDataLoader(train_dataset, shuffle=True)
    test_loader = TestDataLoader(test_dataset)

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

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    params_n = count_parameters(model)
    weights_filename = f"{config['model_name']}-{config['feature']}-{params_n // 1e+6}m-{config['num_epochs']}epoch.pt"


    # set current process to high priority
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows-specific

    if use_wandb: wandb.init(
        project="bhw2",
        config=config
    )

    train_losses, val_losses = train(model=model, 
                                    optimizer=optimizer, 
                                    num_epochs=config['num_epochs'], 
                                    train_loader=train_loader, 
                                    val_loader=test_loader, 
                                    criterion=criterion, 
                                    vocab_trg=vocab_trg, 
                                    scheduler=scheduler, 
                                    use_wandb=use_wandb)

    model.save(weights_filename)

    plot_losses(train_losses, val_losses)