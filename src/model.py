from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, config):
        super(AutoEncoder, self).__init__()
        
        self.encoder = EncoderDecoder(
            in_size       =config['input_size'],
            out_size      =config['hidden_dim'],
            layers_sizes  =config['layers'],
            use_bn        =config['use_bn'],
            activation_cls=config['activation']
        )
        
        self.decoder = EncoderDecoder(
            in_size       =config['hidden_dim'],
            out_size      =config['input_size'],
            layers_sizes  =config['layers'][::-1],
            use_bn        =config['use_bn'],
            activation_cls=config['activation']
        )
    
    def forward(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        return decoded_x, encoded_x


class EncoderDecoder(nn.Sequential):
    def __init__(self, in_size, out_size, layers_sizes, use_bn=True, activation_cls=nn.ReLU):
        layers_sizes = [in_size] + layers_sizes
        layers = []
        
        for i in range(1, len(layers_sizes)):
            layers.append(LinearBNActivation(
                in_size=layers_sizes[i-1],
                out_size=layers_sizes[i],
                use_bn=use_bn,
                activation_cls=activation_cls,
            ))
        
        # Don't apply BatchNorm and activation at last layer
        layers.append(LinearBNActivation(
            in_size=layers_sizes[-1],
            out_size=out_size,
            use_bn=False,
            activation_cls=nn.Identity,
        ))
        
        super(EncoderDecoder, self).__init__(*layers)

        
class LinearBNActivation(nn.Sequential):
    def __init__(self, in_size, out_size, use_bn=True, activation_cls=nn.ReLU):
        layers = [
            nn.Linear(in_size, out_size),
            nn.BatchNorm1d(out_size) if use_bn else nn.Identity(),
            activation_cls(),
        ]
        super(LinearBNActivation, self).__init__(*layers)
