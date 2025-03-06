import os
import subprocess
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

unk_idx, pad_idx, bos_idx, eos_idx = 0, 1, 2, 3


def eval_bleu(pred_filename, ref_filename):
    cmd = f"sacrebleu {ref_filename} --tokenize none --width 2 -b -i {pred_filename}"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout.strip()

    bleu_score = float(output)

    # os.remove(pred_filename)
    # os.remove(ref_filename)
    return bleu_score

# no_unk = set(['<BOS>', '<EOS>', '<PAD>'])
no_unk = set(['<BOS>', '<EOS>'])

def get_bleu(model, dataloader, vocab_trg, device='cuda'):
    pred_filename = "val.pred.en"
    ref_filename = "val.ref.en"

    with open(pred_filename, 'w', encoding='utf-8') as f_pred, \
         open(ref_filename, 'w', encoding='utf-8') as f_ref:
        with torch.no_grad():
            for src, trg in tqdm(dataloader):
                predictions = model.inference(src, max_len=None, device=device) # batch
                for i in range(len(src)):
                    pred_text = vocab_trg.decode(predictions[i], ignore=no_unk)
                    f_pred.write(" ".join(pred_text) + '\n')

                for trg_seq in trg:
                    ref_text = vocab_trg.decode(trg_seq, ignore=no_unk)
                    # ref_text = vocab_trg.decode(trg_seq)
                    f_ref.write(" ".join(ref_text) + '\n')
                  

    bleu_score = eval_bleu(pred_filename, ref_filename)

    return bleu_score


submission_filename = 'submission.en'
def make_submission(model, submission_loader, vocab_trg, submission_filename=submission_filename, device='cuda'):
    with open(submission_filename, 'w', encoding='utf-8') as f_pred:
        with torch.no_grad():
            for a in tqdm(submission_loader):
                predictions = model.inference(a, max_len=None, device=device) # batch
                for i in range(len(a)):
                    pred_text = vocab_trg.decode(predictions[i], ignore=no_unk)
                    f_pred.write(" ".join(pred_text) + '\n')

