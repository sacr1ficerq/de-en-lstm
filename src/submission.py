from dataset2 import bucket_iterator

import subprocess
import torch

from tqdm import tqdm

unk_idx, pad_idx, bos_idx, eos_idx = 0, 1, 2, 3

from time import sleep

from collections import Counter
import numpy as np

from torch.utils.data import Dataset

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

def get_precisions(ref, c, n=4, verbose=False):
    assert isinstance(c, list)
    assert isinstance(ref,list)
    correct = np.zeros(n)
    total = np.zeros(n)
    if len(c) == 0:
        return correct, total

    for n in range(1, n+1):
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

def bleu_from_precision(correct, total, pen, n=4, verbose=False):
    pres = np.zeros(n)
    smooth = 1
    for n in range(n):
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
    p = geomean(pres[pres != 0]) * 100

    result = pen * p
    if verbose:
        print(f'sinergy:\t{p:0.2f}')
        print(f'BP:\t\t{np.round(pen, 3)}')
        print(*np.round(pres*100, 1), sep='/')
    return result, p, pen

def bleu(ref, c, n=4, verbose=False):
    assert isinstance(ref, list)
    assert isinstance(c, list)
    # correct, total = get_precisions(ref, c, verbose=verbose)
    correct, total = get_precisions(ref, c, n=n, verbose=False)
    # print(pres)

    pen = penalty(len(ref), len(c))

    return bleu_from_precision(correct, total, pen, n=n, verbose=verbose)

def eval_bleu(pred_filename, ref_filename):
    cmd = f"sacrebleu {ref_filename} --tokenize none --width 2 -b -i {pred_filename}"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout.strip()

    bleu_score = float(output)

    # os.remove(pred_filename)
    # os.remove(ref_filename)
    return bleu_score

no_unk = set(['<BOS>', '<EOS>', '<PAD>', '<SUB>', '<NUM>'])
# no_unk = set(['<BOS>', '<EOS>', '<NUM>', '<PAD>'])

def get_bleu(model, dataset, raw_dataset, vocab_trg, vocab_src, use_beam=False, beam_width=5, n=int(5e2), border=0.0, batch_size=128, device='cuda'):
    trainset = torch.utils.data.Subset(dataset, range(min(n, len(dataset))))
    valset = torch.utils.data.Subset(raw_dataset, range(min(n, len(dataset))))
    penalties = 0
    correct = np.zeros(4)
    total = np.zeros(4)
    if n == None: n = len(dataset)
    cnt = n

    len_ref = 0
    len_pred = 0

    with torch.no_grad():
        for batch_idx, (src, trg) in enumerate(tqdm(bucket_iterator(trainset, batch_size=batch_size, shuffle=False))):
            if cnt == 0:
                break
            if use_beam:
                predictions = model.inference_beam(src, beam_width=beam_width, border=border, device=device) # batch
            else:
                predictions = model.inference(src, device=device) # batch
            for i in range(len(src)):
                if cnt == 0:
                    break
                # print(batch_idx * batch_size + i)
                ref = valset[batch_idx * batch_size + i]
                # print(vocab_trg.decode(trg[i]), ref)
                pred_text = vocab_trg.decode(predictions[i], ignore=no_unk, src=ref, vocab_src=vocab_src)
                penalties += penalty(len(ref), len(pred_text))
                c, t = get_precisions(ref, pred_text)
                correct += c
                total += t
                len_ref += (trg[i] != pad_idx).sum(dim=-1).item()
                len_pred += len(pred_text)

                # sleep(3)
                cnt -= 1

    print('correct:\t', *map(int, correct))
    print('total:\t\t', *map(int, total))
    bp = penalty(len_ref, len_pred)

    return np.round(bleu_from_precision(correct, total, bp, verbose=True), 2)


def bad_bleu(model, dataloader, vocab_trg, vocab_src, raw_dataset, use_beam=False, beam_width=5, n=5e2, border=0.0, device='cuda'):
    src_bad = '../submission/bad_bleu.de'
    trg_bad = '../submission/bad_bleu.en'
    pred_bad = '../submission/bad_bleu_pred.en'
    pred_beam = '../submission/bad_bleu_beam.en'


    print('calculating bleu...')
    penalties = 0
    correct = np.zeros(4)
    total = np.zeros(4)
    if n == None: n = len(raw_dataset)
    cnt = n

    len_ref = 0
    len_pred = 0

    with torch.no_grad():
        with open (src_bad, 'w', encoding='utf-8') as f_src, open(trg_bad, 'w', encoding='utf-8') as f_trg, open(pred_bad, 'w', encoding='utf-8') as f_pred, open(pred_beam, 'w', encoding='utf-8') as f_beam:
            for batch_idx, (src, trg) in enumerate(tqdm(dataloader)):
                predictions_beam = model.inference_beam(src, beam_width=beam_width, border=border, device=device) # batch
                predictions = model.inference(src, device=device) # batch
                for i in range(len(src)):
                    if cnt == 0:
                        break
                    ref = raw_dataset[batch_idx * dataloader.batch_size + i]
                    # print(vocab_trg.decode(trg[i]), ref)
                    pred_text = vocab_trg.decode(predictions[i], ignore=no_unk, src=ref, vocab_src=vocab_src)
                    pred_beam = vocab_trg.decode(predictions_beam[i], ignore=no_unk, src=ref, vocab_src=vocab_src)
                    penalties += penalty(len(ref), len(pred_text))
                    c, t = get_precisions(ref, pred_text)
                    b_beam = bleu(ref, pred_beam)
                    b = bleu(ref, pred_text)
                    if (b - b_beam) * len(pred_text) > 10 * 10:
                        f_src.write(" ".join(vocab_src.decode(src[i], ignore=set(['<EOS>', '<BOS>']), src=ref, vocab_src=vocab_trg)) + '\n')
                        f_trg.write(" ".join(ref) + '\n')
                        f_pred.write(" ".join(pred_text) + '\n')
                        f_beam.write(" ".join(pred_beam) + '\n')
                    correct += c
                    total += t
                    len_ref += list(trg[i]).index(eos_idx) - 1
                    len_pred += len(pred_text)

                    cnt -= 1

    print('correct:\t', *map(int, correct))
    print('total:\t\t', *map(int, total))
    bp = penalty(len_ref, len_pred)
    result = np.round(bleu_from_precision(correct, total, bp, verbose=True), 2)

    return result


def make_submission(model, submission_loader, vocab_trg, vocab_src, filenames, raw_dataset, use_beam=False, beam_width=5, device='cuda'):
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
                    pred_text = vocab_trg.decode(predictions[i], ignore=no_unk, src=t, vocab_src=vocab_src)
                    f_pred.write(" ".join(pred_text) + '\n')