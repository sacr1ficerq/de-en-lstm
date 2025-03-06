from build import build

from config import filenames

device = 'cuda'

config = {
    'model_name': 'LSTM_2',
    'feature': 'regularized',
    'max_len': 48,
    'min_freq_src': 4,
    'min_freq_trg': 4,

    'embedding_dim': 128,
    'hidden_size': 256,
    'num_epochs': 15,
    'weight_decay': 1e-5,
    'label_smoothing': 0.1,
    'dropout': 0.2,

    'learning_rate': 1e-3,
    'gamma': 0.2,
    'patience': 0,
    'threshold': 0.001
}

build(config=config, 
      filenames=filenames, 
      use_wandb=True,
      device=device)