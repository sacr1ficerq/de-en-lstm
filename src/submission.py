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
    return np.exp(1 - (r / c))


def get_precisions(ref, c):
    res = []
    total = np.zeros(4)
    smooth = 1
    for n in range(1, 5):
        c_ngrams = list(zip(*[c[i:] for i in range(n)]))
        c_ngram_cnt = Counter(c_ngrams)

        ref_ngrams = list(zip(*[ref[i:] for i in range(n)]))
        ref_ngram_cnt = Counter(ref_ngrams)

        match_ngrams = 0
        for ngram in c_ngram_cnt:
            if ngram in ref_ngram_cnt:
                match_ngrams += min(c_ngram_cnt[ngram], ref_ngram_cnt[ngram])

        total[n - 1] = sum(c_ngram_cnt.values())
        # precision = match_ngrams / total_c_ngrams if total_c_ngrams > 0 else -1
        if match_ngrams!= 0:
            precision = match_ngrams / total[n - 1] if total[n - 1] != 0 else 0
        else:
            smooth *= 2
            precision = 100 / (smooth * total[n - 1]) / 100
        
        res.append(precision)
    return np.array(res)

def bleu(ref, c):
    pres = get_precisions(ref, c)
    # print(pres)

    # print(pres * 100)

    pen = penalty(len(ref), len(c))
    # print(pen)
    p = geomean(pres[pres != -1])

    return float(np.round(100*p*pen , 2))

def eval_bleu(pred_filename, ref_filename):
    cmd = f"sacrebleu {ref_filename} --tokenize none --width 2 -b -i {pred_filename}"
    print(cmd)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout.strip()

    bleu_score = float(output)

    # os.remove(pred_filename)
    # os.remove(ref_filename)
    return bleu_score

# no_unk = set(['<BOS>', '<EOS>', '<PAD>'])
no_unk = set(['<BOS>', '<EOS>', '<NUM>'])

def get_bleu(model, dataloader, vocab_trg, filenames, raw_dataset, use_beam=False, beam_width=5, device='cuda'):

    pred_filename = filenames['test_pred']
    ref_filename = filenames['test_trg']
    total_penalties = 0
    total_bleu = 0
    pres = np.zeros(4)
    pres_cnt = np.zeros(4)

    with open(pred_filename, 'w', encoding='utf-8') as f_pred,  open(filenames['submission_trg'], 'w', encoding='utf-8') as f_val:
        with torch.no_grad():
            for batch_idx, (src, trg) in enumerate(tqdm(dataloader)):
                if use_beam:
                    predictions = model.inference_beam(src, beam_width=beam_width, device=device) # batch
                else:
                    predictions = model.inference(src, device=device) # batch
                for i in range(len(src)):
                    ref = raw_dataset[batch_idx * dataloader.batch_size + i]
                    # print(vocab_trg.decode(trg[i]), ref)
                    pred_text = vocab_trg.decode(predictions[i], ignore=no_unk, src=ref)
                    total_penalties += penalty(len(ref), len(pred_text))
                    p = get_precisions(ref, pred_text)
                    pres[p != -1] += p[p != -1]
                    pres_cnt += 1


                    total_bleu += bleu(ref, pred_text)
                    f_pred.write(" ".join(pred_text) + '\n')
                    f_pred.flush()
                    f_val.write(" ".join(ref) + '\n')
                    f_val.flush()
                    # print(ref, pred_text, bleu(ref, pred_text))

                    # sleep(3)

                # for trg_seq in trg:
                #     ref_text = vocab_trg.decode(trg_seq, ignore=no_unk)
                #     # ref_text = vocab_trg.decode(trg_seq)
                #     f_ref.write(" ".join(ref_text) + '\n')
    
    print('precisions:', pres / pres_cnt * 100)
    print('precisions_cnt:', pres_cnt)

    print('avg bleu:', geomean(pres/pres_cnt) * total_penalties / len(raw_dataset))

    print('penalties:', total_penalties / len(raw_dataset))
    print('bleu4:', total_bleu / len(raw_dataset))
    bleu_score = eval_bleu(pred_filename, ref_filename)

    return bleu_score


def make_submission(model, submission_loader, vocab_trg, filenames, raw_dataset, use_beam=False, beam_width=5, device='cuda'):
    with open(filenames['submission_trg'], 'w', encoding='utf-8') as f_pred:
        with torch.no_grad():
            for batch_idx, a in enumerate(tqdm(submission_loader)):
                predictions = model.inference(a, max_len=-1, device=device) # batch
                if not use_beam:
                    predictions = model.inference(a, max_len=-1, device=device) # batch
                else:
                    predictions = model.inference_beam(a, max_len=-1, beam_width=beam_width, device=device) # batch
                for i in range(len(a)):
                    t = None
                    if raw_dataset:
                        t = raw_dataset[batch_idx * submission_loader.batch_size + i]
                        # print(t)
                    pred_text = vocab_trg.decode(predictions[i], ignore=no_unk, src=t)
                    f_pred.write(" ".join(pred_text) + '\n')