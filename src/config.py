folders = {
    'extract_to': '../'
}

folders['data'] = folders['extract_to'] + 'data/'
folders['weights'] = '../weights/'

filenames = {
    'train_src': folders['data'] + 'train.de-en.de',
    'train_trg': folders['data'] + 'train.de-en.en',
    'test_src': folders['data'] + 'val.de-en.de',
    'test_trg': folders['data'] + 'val.de-en.en',
    'submission_src': folders['data'] + 'test1.de-en.de',
}