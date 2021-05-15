import torch
from scipy.stats import truncnorm


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
        torch.nn.init.xavier_uniform_(self.conv.weight.data)
        torch.nn.init.constant_(self.conv.bias.data, 0.0)

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
