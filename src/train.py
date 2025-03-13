
from lstm3 import LSTM_3

from submission import get_bleu
from dataset2 import TranslationDataset, Vocab, bucket_iterator, RawDataset, TrainDataLoader

from tqdm import tqdm

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np


def train_epoch(model, optimizer, train_dataset, val_dataset, criterion, vocab_trg, batch_size=128, teacher_forcing=1):
    model.train()
    total_train_loss = 0

    for src_seq, trg_seq in tqdm(bucket_iterator(train_dataset, batch_size=batch_size)):
        trg_input = trg_seq[:, :-1]
        trg_output = trg_seq[:, 1:]

        logits = model(src_seq, trg_input, teacher_forcing=teacher_forcing)
        # logits = model.forward_no_tf(src_seq, trg_input)
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
    total_val_loss_no_tf = 0
    with torch.no_grad():
        for src_seq, trg_seq in tqdm(bucket_iterator(val_dataset, batch_size=batch_size)):
            trg_input = trg_seq[:, :-1]
            trg_output = trg_seq[:, 1:]

            logits = model(src_seq, trg_input, teacher_forcing=1.0)
            logits = logits.view(-1, len(vocab_trg))
            trg_output = trg_output.reshape(-1)

            loss = criterion(logits, trg_output)
            total_val_loss += loss.item()

        for src_seq, trg_seq in tqdm(bucket_iterator(val_dataset, batch_size=batch_size)):
            trg_input = trg_seq[:, :-1]
            trg_output = trg_seq[:, 1:]

            logits = model(src_seq, trg_input, teacher_forcing=0.0)
            logits = logits.view(-1, len(vocab_trg))
            trg_output = trg_output.reshape(-1)

            loss = criterion(logits, trg_output)
            total_val_loss_no_tf += loss.item()

    avg_train_loss = total_train_loss / (len(train_dataset) // batch_size + 1)
    avg_val_loss = total_val_loss / (len(val_dataset) // batch_size + 1)
    return avg_train_loss, avg_val_loss

def train(config, filenames, folders, device='cuda', use_wandb=False, vocab_src=None, vocab_trg=None, train_dataset=None, val_dataset=None, demonstrate=False):
    bpe_vocab_size = config.get('bpe_vocab_size', 25000)
    if vocab_src == None: vocab_src = Vocab(filenames['train_src'], min_freq=config['min_freq_src'], max_freq_sub=config.get('max_freq_sub', 8))
    if vocab_trg == None: vocab_trg = Vocab(filenames['train_trg'], min_freq=config['min_freq_trg'], max_freq_sub=config.get('max_freq_sub', 8))

    tf_start = None
    tf_decrease = None
    tf_from_epoch = None
    use_tf = config.get('use_tf', False)

    if use_tf:
        tf_start = config['tf_start']
        tf_decrease = config['tf_decrease']        
        tf_from_epoch = config['tf_from_epoch']

    if train_dataset==None: train_dataset = TranslationDataset(vocab_src, 
                                    vocab_trg, 
                                    filenames['train_src'], 
                                    filenames['train_trg'], 
                                    max_len=config['max_len'], 
                                    device=device,
                                    sort_lengths=False)

    if val_dataset==None: val_dataset = TranslationDataset(vocab_src, 
                                    vocab_trg, 
                                    filenames['test_src'], 
                                    filenames['test_trg'], 
                                    device=device, 
                                    sort_lengths=True)

    unk_idx, pad_idx, bos_idx, eos_idx = 0, 1, 2, 3

    batch_size=config.get('batch_size', 128)

    config['src_vocab_size'] = len(vocab_src)
    config['trg_vocab_size'] = len(vocab_trg)

    model = LSTM_3(config=config).to(device)

    train_loader = TrainDataLoader(train_dataset, 256, True)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, 
                                    label_smoothing=config['label_smoothing'])

    optimizer = optim.AdamW(model.parameters(), 
                            lr=config['learning_rate'], 
                            weight_decay=config['weight_decay'],
                            amsgrad=config.get('amsgrad', False),
                            foreach=True)
    
    scheduler = ReduceLROnPlateau(optimizer, 
                                patience=config['patience'], 
                                factor=config['gamma'], 
                                threshold=config['threshold'],
                                cooldown=config.get('cooldown', 1))
    
    raw_dataset_val = RawDataset(filenames['test_trg'])
    raw_dataset_val.src = [raw_dataset_val.src[i] for i in val_dataset.ids]
    # raw_dataset_train = RawDataset(filenames['train_trg'])
    # raw_dataset_train.src = [raw_dataset_train.src[i] for i in train_dataset.ids[:int(5e2)+1]]


    print(model)
    params_n = model.count_parameters()
    print(f'Parameters: {params_n//1e+3}k')
    config['parameters'] = params_n
    print(criterion)
    print(scheduler.state_dict())
    print(optimizer)

    weights_filename = folders['weights'] + f"{config['model_name']}-{config['feature']}-{params_n // 1e+6}m-{config['num_epochs']}epoch.pt"

    # bleu, sinerrgy, bp = get_bleu(model, train_dataset, raw_dataset_train, vocab_trg, vocab_src)

    if use_wandb: wandb.init(
        project="bhw2",
        config=config
    )

    # if use_wandb: wandb.watch(model, log="parameters", log_freq=10)
    train_losses = []
    val_losses = []

    teacher_forcing = 1.0
    model.demonstrate(train_dataset, vocab_src, vocab_trg, examples=5, device=device, wait=0, verbose=0)
    for epoch in range(1, config['num_epochs']+1):
        if use_tf:
            if epoch >= tf_from_epoch:
                teacher_forcing = tf_start  - tf_decrease * (epoch - tf_from_epoch)

        train_loss, val_loss = train_epoch(model, 
                                            optimizer, 
                                            train_dataset, 
                                            val_dataset, 
                                            criterion, 
                                            vocab_trg, 
                                            teacher_forcing=teacher_forcing)
        
        val_losses.append(train_loss)
        train_losses.append(val_loss)

        if use_wandb: 
            wandb.log({"train_loss": train_loss,
                       "val_loss": val_loss,
                       "epoch": epoch,})
            wandb.log({"lr": scheduler.get_last_lr()[0]})

        if scheduler:
            scheduler.step(val_loss)
            print('lr:', scheduler.get_last_lr()[0])

        if epoch >= 6:
            checkpoint_path = f'lstm-save-{epoch}.pt'
            model.save(checkpoint_path, folders['saves'])
            # if use_wandb: wandb.save(checkpoint_path)
        elif config.get('lr_manual_decrease', False):
            # adjust learning rate dynamically
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8

        model.demonstrate(train_dataset, vocab_src, vocab_trg, examples=5, device=device, wait=0, verbose=0)

        # train_bleu, train_sinergy, train_bp  = get_bleu(model, train_dataset, raw_dataset_train, vocab_trg, vocab_src)
        val_bleu, val_sinergy, val_bp = get_bleu(model, val_dataset, raw_dataset_val, vocab_trg, vocab_src)
        if use_wandb: wandb.log({
                                # "Train BLEU4": train_bleu,
                                #  "Train Sinergy": train_sinergy,
                                #  "Train BP": train_bp,

                                 "BLEU4": val_bleu,
                                 "Sinergy": val_sinergy,
                                 "BP": val_bp})

        if demonstrate: model.demonstrate(train_dataset, vocab_src, vocab_trg, examples=3, device=device, wait=0)

        print(f"Epoch [{epoch}/{config['num_epochs']}]\tTrain Loss: {train_loss:.4f}\tVal Loss: {val_loss:.4f}")
        # print(f"Train BLUE4: {train_bleu}\tTrain Sinergy: {train_sinergy}\tTrain BP: {train_bp}")
        print(f"BLUE4: {val_bleu}\tSinergy: {val_sinergy}\tBP: {val_bp}")

    model.train_loss = np.array(train_losses)
    model.val_loss = np.array(val_losses)

    model.save(weights_filename, folders['weights'])
    wandb.finish()

    return train_losses, val_losses