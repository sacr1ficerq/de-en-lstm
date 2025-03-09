import os
import subprocess
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

unk_idx, pad_idx, bos_idx, eos_idx = 0, 1, 2, 3

from time import sleep

from collections import Counter
import numpy as np

def geomean(a):
    a = np.array(a)
    # a[a==0] = 0.1
    log_sum = np.sum(np.log(a))
    geometric_mean = np.exp(log_sum / len(a))
    return geometric_mean

def penalty(r, c):
    if c > r: return 1
    if c == 0:
        return 1e-10
    return np.exp(1 - (r / c))


def get_precisions(ref, c, verbose=False):
    assert isinstance(c, list)
    assert isinstance(ref,list)
    correct = np.zeros(4)
    total = np.zeros(4)
    if len(c) == 0:
        return correct, total

    for n in range(1, 5):
        c_ngrams = list(zip(*[c[i:] for i in range(n)]))

        c_ngram_cnt = Counter(c_ngrams)

        ref_ngrams = list(zip(*[ref[i:] for i in range(n)]))
        ref_ngram_cnt = Counter(ref_ngrams)

        for ngram in c_ngram_cnt:
            if ngram in ref_ngram_cnt:
                correct[n - 1] += min(c_ngram_cnt[ngram], ref_ngram_cnt[ngram])

        total[n - 1] = sum(c_ngram_cnt.values())
        if verbose: print(f'{n}: {int(correct[n-1])}/{int(total[n-1])}')
        
    return correct, total

def bleu_from_precision(correct, total, pen, verbose=False):
    pres = np.zeros(4)
    smooth = 1
    for n in range(4):
        precision = 0
        if total[n] == 0:
            precision = 0
        else:
            if correct[n] != 0:
                precision = correct[n] / total[n]
            else:
                smooth *= 2
                precision = 1 / (smooth * total[n])
        pres[n] = precision
    p = geomean(pres[pres != 0]) 

    if verbose:
        print(f'sinergy:\t{100*p:0.2f}')
        print(f'BP:\t\t{np.round(pen, 3)}')
        print(*np.round(pres*100, 1), sep='/')
    return pen * p * 100

def bleu(ref, c, verbose=False):
    assert isinstance(ref, list)
    assert isinstance(c, list)
    # correct, total = get_precisions(ref, c, verbose=verbose)
    correct, total = get_precisions(ref, c, verbose=False)
    # print(pres)

    pen = penalty(len(ref), len(c))

    return bleu_from_precision(correct, total, pen, verbose=verbose)

def eval_bleu(pred_filename, ref_filename):
    cmd = f"sacrebleu {ref_filename} --tokenize none --width 2 -b -i {pred_filename}"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout.strip()

    bleu_score = float(output)

    # os.remove(pred_filename)
    # os.remove(ref_filename)
    return bleu_score

no_unk = set(['<BOS>', '<EOS>', '<PAD>'])
# no_unk = set(['<BOS>', '<EOS>', '<NUM>', '<PAD>'])

def get_bleu(model, dataloader, vocab_trg, raw_dataset, use_beam=False, beam_width=5, n=None, border=0.0, device='cuda'):
    penalties = 0
    correct = np.zeros(4)
    total = np.zeros(4)
    if n == None: n = len(raw_dataset)
    cnt = n

    len_ref = 0
    len_pred = 0

    with torch.no_grad():
        for batch_idx, (src, trg) in enumerate(tqdm(dataloader)):
            if use_beam:
                predictions = model.inference_beam(src, beam_width=beam_width, border=border, device=device) # batch
            else:
                predictions = model.inference(src, device=device) # batch
            for i in range(len(src)):
                ref = raw_dataset[batch_idx * dataloader.batch_size + i]
                # print(vocab_trg.decode(trg[i]), ref)
                pred_text = vocab_trg.decode(predictions[i], ignore=no_unk, src=ref)
                penalties += penalty(len(ref), len(pred_text))
                c, t = get_precisions(ref, pred_text)
                correct += c
                total += t
                len_ref += list(trg[i]).index(eos_idx) - 1
                len_pred += len(pred_text)

                # sleep(3)
                cnt -= 1
                if cnt == 0:
                    break
            if cnt == 0:
                break

    print('correct:\t', *map(int, correct))
    print('total:\t\t', *map(int, total))
    bp = penalty(len_ref, len_pred)
    bleu = np.round(bleu_from_precision(correct, total, bp, verbose=True), 2)

    return bleu


def make_submission(model, submission_loader, vocab_trg, filenames, raw_dataset, use_beam=False, beam_width=5, device='cuda'):
    with open(filenames['submission_trg'], 'w', encoding='utf-8') as f_pred:
        with torch.no_grad():
            for batch_idx, a in enumerate(tqdm(submission_loader)):
                predictions = model.inference(a, device=device) # batch
                if not use_beam:
                    predictions = model.inference(a, device=device) # batch
                else:
                    predictions = model.inference_beam(a, beam_width=beam_width, device=device) # batch
                for i in range(len(a)):
                    t = raw_dataset[batch_idx * submission_loader.batch_size + i]
                        # print(t)
                    pred_text = vocab_trg.decode(predictions[i], ignore=no_unk, src=t)
                    f_pred.write(" ".join(pred_text) + '\n')