from torch import nn


model_config = {
    'input_size': 4,
    'hidden_dim': 3,
    'layers': [200, 100, 50],
    'activation': nn.Tanh,
    'use_bn': False,
    
    'n_epochs': 50,
    'weight_decay': 0.1,
    'batch_size': 4096,
    'lr': 1e-2,
    'lr_step_size': 9999999,     # no lr scheduler
    'lr_gamma': 0.5,
}