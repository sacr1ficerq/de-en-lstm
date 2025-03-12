
from lstm3 import LSTM_3

from submission import get_bleu
from dataset2 import TranslationDataset, Vocab, TrainDataLoader, TestDataLoader, RawDataset

from tqdm import tqdm

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

def train_epoch(model, optimizer, train_loader, val_loader, criterion, vocab_trg, teacher_forcing=1):
    model.train()
    total_train_loss = 0

    for src_seq, trg_seq in tqdm(train_loader):
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
        for src_seq, trg_seq in tqdm(val_loader):
            trg_input = trg_seq[:, :-1]
            trg_output = trg_seq[:, 1:]

            logits = model(src_seq, trg_input, teacher_forcing=1.0)
            logits = logits.view(-1, len(vocab_trg))
            trg_output = trg_output.reshape(-1)

            loss = criterion(logits, trg_output)
            total_val_loss += loss.item()

        for src_seq, trg_seq in tqdm(val_loader):
            trg_input = trg_seq[:, :-1]
            trg_output = trg_seq[:, 1:]

            logits = model(src_seq, trg_input, teacher_forcing=0.0)
            logits = logits.view(-1, len(vocab_trg))
            trg_output = trg_output.reshape(-1)

            loss = criterion(logits, trg_output)
            total_val_loss_no_tf += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_loss_no_tf = total_val_loss_no_tf / len(val_loader)

    return avg_train_loss, avg_val_loss, avg_val_loss_no_tf

def train(config, filenames, folders, device='cuda', use_wandb=False, vocab_src=None, vocab_trg=None, train_dataset=None, val_dataset=None, demonstrate=False):
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
                                    device=device)

    if val_dataset==None: val_dataset = TranslationDataset(vocab_src, 
                                    vocab_trg, 
                                    filenames['test_src'], 
                                    filenames['test_trg'], 
                                    max_len=72, 
                                    device=device, 
                                    sort_lengths=False)

    unk_idx, pad_idx, bos_idx, eos_idx = 0, 1, 2, 3

    train_loader = TrainDataLoader(train_dataset, batch_size=config.get('batch_size', 128), shuffle=True)
    val_loader = TestDataLoader(val_dataset, batch_size=256, shuffle=False)

    config['src_vocab_size'] = len(vocab_src)
    config['trg_vocab_size'] = len(vocab_trg)

    model = LSTM_3(config=config).to(device)

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

    raw_dataset = RawDataset(filenames['test_trg'])

    print(model)
    params_n = model.count_parameters()
    print(f'Parameters: {params_n//1e+3}k')
    config['parameters'] = params_n
    print(criterion)
    print(scheduler)
    print(optimizer)

    weights_filename = folders['weights'] + f"{config['model_name']}-{config['feature']}-{params_n // 1e+6}m-{config['num_epochs']}epoch.pt"

    bleu_score = get_bleu(model, val_loader, vocab_trg, vocab_src, raw_dataset)
    # bleu_score_beam = get_bleu(model, val_loader, vocab_trg, vocab_src, raw_dataset, use_beam=True)

    # model.demonstrate(train_loader, vocab_src, vocab_trg, examples=5, device=device, wait=0)
    # print(f"Start BLEU4: {bleu_score}")
    # print(f"Start BLEU4-beam: {bleu_score_beam}")

    if use_wandb: wandb.init(
        project="bhw2",
        config=config
    )

    # if use_wandb: wandb.watch(model, log="parameters", log_freq=10)
    train_losses = []
    val_losses = []

    teacher_forcing = 1.0

    for epoch in range(1, config['num_epochs']+1):
        if use_tf:
            if epoch >= tf_from_epoch:
                teacher_forcing = tf_start  - tf_decrease * (epoch - tf_from_epoch)

        train_loss, val_loss, val_loss_no_tf = train_epoch(model, 
                                            optimizer, 
                                            train_loader, 
                                            val_loader, 
                                            criterion, 
                                            vocab_trg, 
                                            teacher_forcing=teacher_forcing)
        
        val_losses.append(train_loss)
        train_losses.append(val_loss)

        if use_wandb: 
            wandb.log({"train_loss": train_loss,
                       "val_loss": val_loss,
                       "val_loss_no_tf":  val_loss_no_tf,
                       "epoch": epoch,})
            wandb.log({"lr": scheduler.get_last_lr()[0]})

        if scheduler:
            scheduler.step(val_loss)
            print('lr:', scheduler.get_last_lr()[0])



        if epoch >= 6:
            checkpoint_path = f'lstm-save-{epoch}.pt'
            model.save(checkpoint_path, folders['saves'])
            if epoch % 2 == 0:
                bleu_score_beam = get_bleu(model, val_loader, vocab_trg, vocab_src, raw_dataset, use_beam=True)
                print(f"BLEU4-beam: {bleu_score_beam}")
            # if use_wandb: wandb.save(checkpoint_path)
        elif config.get('lr_manual_decrease', False):
            # adjust learning rate dynamically
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8

        bleu_score = get_bleu(model, val_loader, vocab_trg, vocab_src, raw_dataset)
        if use_wandb: wandb.log({"BLEU4": bleu_score})

        if epoch % 2 == 0:
            if demonstrate: model.demonstrate(train_loader, vocab_src, vocab_trg, examples=5, device=device, wait=0)

        print(f"Epoch [{epoch}/{config['num_epochs']}]\tTrain Loss: {train_loss:.4f}\tVal Loss: {val_loss:.4f}\tBLEU4: {bleu_score}")


    model.train_loss = np.array(train_losses)
    model.val_loss = np.array(val_losses)

    model.save(weights_filename, folders['weights'])
    wandb.finish()

    return train_losses, val_losses