folders = {
    'extract_to': '../'
}

folders['data'] = folders['extract_to'] + 'data/'
folders['weights'] = '../weights/'
folders['saves'] = '../weights/saves/'
folders['submission'] = '../submission/'

filenames = {
    'train_src': folders['data'] + 'train.de-en.de',
    'train_trg': folders['data'] + 'train.de-en.en',
    'test_src': folders['data'] + 'val.de-en.de',
    'test_trg': folders['data'] + 'val.de-en.en',
    'test_pred': folders['submission'] + 'val_pred.en',
    'submission_src': folders['data'] + 'test1.de-en.de',
    'submission_trg': folders['submission'] + 'submission.en',
    'low_bleu_src': folders['submission'] + 'low_bleu.de',
    'low_bleu_trg': folders['submission'] + 'low_bleu.en'
}