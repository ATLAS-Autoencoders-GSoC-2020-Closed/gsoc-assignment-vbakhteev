from torch import nn


model_config = {
    'input_size': 4,
    'hidden_dim': 3,
    'layers': [100, 50, 50, 20],
    'activation': nn.ELU,
    'use_bn': False,
    
    'n_epochs': 50,
    'weight_decay': 0.001,
    'batch_size': 1024,
    'lr': 1e-2,
    'lr_step_size': 10,
    'lr_gamma': 0.5,
}