import torch
from scipy.stats import truncnorm
from torch._C import device
from torch import nn


def truncated_normal_(tensor, mean=0.0, std=1.0):
    # PT doesn't have truncated normal.
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/18
    values = truncnorm.rvs(-2, 2, size=tensor.shape)
    values = mean + std * values
    tensor.copy_(torch.from_numpy(values))
    return tensor


def fc_init_(module):
    if hasattr(module, 'weight') and module.weight is not None:
        truncated_normal_(module.weight.data, mean=0.0, std=0.01)
    if hasattr(module, 'bias') and module.bias is not None:
        torch.nn.init.constant_(module.bias.data, 0.0)
    return module


class LinearBlock(torch.nn.Module):

    """ Linear block after feature extraction
    Arguments:-
      input_size, output_size """

    def __init__(self, input_size, output_size):
        super(LinearBlock, self).__init__()
        self.relu = torch.nn.ReLU()
        self.normalize = torch.nn.BatchNorm1d(
            output_size,
            affine=True,
            momentum=0.999,
            eps=1e-3,
            track_running_stats=False,
        )
        self.linear = torch.nn.Linear(input_size, output_size)
        fc_init_(self.linear)

    def forward(self, x):
        x = self.linear(x)
        x = self.normalize(x)
        x = self.relu(x)
        return x


class ConvBlock(torch.nn.Module):

    """ Convolutional Block consisting of Conv - BatchNorm - ReLU - MaxPool(1/0)
    Arguments:-
      in_channels: no of channels input
      out_channels: no of filters/kernels
      kernel_size
      max_pool: bool
      stride: if max_pool == T -> max_pool_stride else -> conv_stride
    """

    def __init__(self, in_channels, out_channels, kernel_size, max_pool, stride):
        super(ConvBlock, self).__init__()

        if max_pool:
            self.max_pool = torch.nn.MaxPool2d(
                kernel_size=stride, stride=stride)
            stride = (1, 1)
        else:
            self.max_pool = Lambda(lambda x: x)

        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, padding=1, stride=stride)
        self.norm = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU()
        torch.nn.init.uniform_(self.norm.weight)
#         torch.nn.init.xavier_uniform_(self.conv.weight.data)
#         torch.nn.init.constant_(self.conv.bias.data, 0.0)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ConvBase(torch.nn.Sequential):
    """ Creates blocks of stacked ConvBlock's 
    Arguments: 
        channels: no of input channels in the 1st ConvBlock
        hidden: no of channels in hidden blocks 
        layers: no of ConvBlock's
        max_pool, stride: params as per ConvBlock """

    def __init__(self, channels, hidden, layers, max_pool, stride, out_channels=None):
        if out_channels == None:
            out_channels = hidden
        block = [ConvBlock(in_channels=channels, out_channels=hidden, kernel_size=(3, 3),
                           max_pool=max_pool, stride=stride)]
        for _ in range(layers-2):
            block.append(ConvBlock(in_channels=hidden, out_channels=hidden, kernel_size=(3, 3),
                         max_pool=max_pool, stride=stride))
        block.append(ConvBlock(in_channels=hidden, out_channels=out_channels, kernel_size=(3, 3),
                               max_pool=max_pool, stride=stride))

        super(ConvBase, self).__init__(*block)


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.lamb = func

    def forward(self, * args, **kwargs):
        return self.lamb(* args, **kwargs)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class OmniCNN(torch.nn.Module):
    def __init__(self, output_size, stride, hidden_size=64, layers=4):
        super(OmniCNN, self).__init__()
        self.hidden_size = hidden_size
        self.base = ConvBase(channels=1,
                             hidden=self.hidden_size,
                             layers=layers, max_pool=False, stride=stride
                             )
        self.features = torch.nn.Sequential(
            Lambda(lambda x: x.view(-1, 1, 28, 28)),
            self.base,
            Lambda(lambda x: x.mean(dim=[2, 3])),
            Flatten(),
        )
        self.classifier = torch.nn.Linear(
            self.hidden_size, output_size, bias=True)
        self.classifier.weight.data.normal_()
        self.classifier.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MiniImageCNN(torch.nn.Module):
    def __init__(
        self,
        output_size,
        stride,
        hidden_size=32,
        layers=4,
        channels=3,
        embedding_size=None
    ):
        super(MiniImageCNN, self).__init__()
        if embedding_size is None:
            embedding_size = 25 * hidden_size
        base = ConvBase(
            hidden=hidden_size,
            channels=channels,
            max_pool=True,
            layers=layers,
            stride=stride,
        )
        self.features = torch.nn.Sequential(
            base,
            Flatten(),
        )
        self.classifier = torch.nn.Linear(
            embedding_size,
            output_size,
            bias=True,
        )
        torch.nn.init.xavier_uniform_(self.classifier.weight.data)
        torch.nn.init.constant_(self.classifier.bias.data, 0.0)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class EncoderNN(torch.nn.Module):
    def __init__(self, channels, stride, max_pool, hidden_size=64, layers=4):
        super(EncoderNN, self).__init__()

        self.hidden_size = hidden_size
        self.base = ConvBase(channels=channels,
                             hidden=self.hidden_size,
                             layers=layers, max_pool=max_pool, stride=stride
                             )
        self.encoder = torch.nn.Sequential(self.base, Flatten())

    def forward(self, x):
        x = self.encoder(x)
        return x


class BidirLSTM(torch.nn.Module):
    def __init__(self, size, layers):
        """Bidirectional LSTM used to generate fully conditional embeddings (FCE) of the support set.
        # Arguments:-
        size: Size of input and hidden layers (must be same to enable skip-connections)
        layers: Number of LSTM layers
        """
        super(BidirLSTM, self).__init__()
        self.num_layers = layers
        self.batch_size = 1

        self.lstm = torch.nn.LSTM(input_size=size,
                                  num_layers=layers,
                                  hidden_size=size,
                                  bidirectional=True)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x, None)
        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]
        # Skip connection between inputs and outputs
        output = forward_output + backward_output + x
        return output, hn, cn


class attLSTM(torch.nn.Module):
    def __init__(self, size, unrolling_steps):
        """Attentional LSTM used to generate fully conditional embeddings (FCE) of the query set.
        # Arguments:-
        size: Size of input and hidden layers (must be same to enable skip-connections)
        unrolling_steps: Number of steps of attention over the support set to compute. Analogous to number of
            layers in a regular LSTM
        """
        super(attLSTM, self).__init__()
        self.unrolling_steps = unrolling_steps
        self.lstm_cell = torch.nn.LSTMCell(input_size=size,
                                           hidden_size=size)

    def forward(self, support, queries, device):

        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]

        h_hat = torch.zeros_like(queries).to(device)
        c = torch.zeros(batch_size, embedding_dim).to(device)

        for k in range(self.unrolling_steps):
            h = h_hat + queries
            attentions = torch.mm(h, support.t())
            attentions = attentions.softmax(dim=1)
            readout = torch.mm(attentions, support)
            h_hat, c = self.lstm_cell(queries, (h + readout, c))

        h = h_hat + queries

        return h


class MatchingNetwork(torch.nn.Module):
    def __init__(self, num_input_channels, stride, max_pool,
                 lstm_layers, lstm_input_size, unrolling_steps, device):
        """Creates a Matching Network.
        # Arguments:-
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            stride: stride for EncodderNN
            max_pool: bool for EncodderNN
            lstm_layers: Number of LSTM layers in the bidrectional LSTM g that embeds the support set (fce = True)
            lstm_input_size: Input size for the bidirectional and Attention LSTM. This is determined by the embedding
                dimension of the few shot encoder which is in turn determined by the size of the input data. Hence we
                have Omniglot -> 64, miniImageNet -> 1600.
            unrolling_steps: Number of unrolling steps to run the Attention LSTM
            device: Device on which to run computation
        """
        super(MatchingNetwork, self).__init__()
        self.stride = stride
        self.max_pool = max_pool
        self.num_input_channels = num_input_channels
        self.encoder = EncoderNN(
            self.num_input_channels, self.stride, self.max_pool).to(device)
        self.support_encoder = BidirLSTM(
            lstm_input_size, lstm_layers).to(device)
        self.query_encoder = attLSTM(
            lstm_input_size, unrolling_steps=unrolling_steps).to(device)

    def forward(self, x):
        pass

# class Siamese(torch.nn.Module):
#     def __init__(self, ):
#         super(Siamese).__init__()


