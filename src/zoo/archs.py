import torch
from scipy.stats import truncnorm
from torch._C import device
from torch import nn
from torch.nn import functional as F
from torch.utils.data import dataset


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


class LEncoder(nn.Module):
    """ Linear Encoder to transform an input image representation/vector into a latent-space gaussian distribution parametrized by its mean and log-variance. """

    def __init__(self,
                 input_dims: int,
                 latent_dim: int,
                 act_fn: object = nn.ReLU):
        """
        Inputs:
            - input_dims : Number of input dimensions of the image-embedding
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()

        self.net = nn.Sequential(nn.Linear(input_dims, 256),
                                 act_fn(),
                                 nn.Linear(256, 128),
                                 act_fn())
        self.h1 = nn.Linear(128, latent_dim)
        self.h2 = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.net(x)
        mu = self.h1(x)
        log_var = self.h2(x)
        return mu, log_var


class LDecoder(nn.Module):
    """ Linear Decoder for reconstructing an image representation/vector using a latent variable z as input. """

    def __init__(self,
                 input_dims: int,
                 latent_dim: int,
                 act_fn: object = nn.LeakyReLU):
        """
        Inputs:
            - input_dims : Number of input dimensions of the image-embedding to be reconstructed
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            act_fn(),
            nn.Linear(128, 256),
            act_fn(),
            nn.Linear(256, input_dims)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class LVAE(nn.Module):
    """ Module for a Linear VAE: Linear Encoder + Decoder. The decoder uses the latent-space vector z 
    drawn from the gaussian distribution and a one-hot encoded vector of class label to reconstruct the input vector. """

    def __init__(self, in_dims, y_shape, latent_dim=64):
        super(LVAE, self).__init__()
        self.in_dims = in_dims
        self.latent_dim = latent_dim
        self.dec_latent_dim = self.latent_dim + y_shape

        self.encoder = LEncoder(input_dims=self.in_dims,
                                latent_dim=self.latent_dim)

        self.decoder = LDecoder(input_dims=self.in_dims,
                                latent_dim=self.dec_latent_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, y):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x = self.decoder(torch.cat([z, y], dim=1))
        return x, mu, log_var


class CEncoder(nn.Module):
    """ Convolutional Encoder to transform an input image into a latent-space gaussian distribution parametrized by its mean and log-variance. """

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 dataset: str,
                 act_fn: object = nn.ReLU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers use 2x of it.
            - latent_dim : Dimensionality of latent representation z
            - dataset: name of the dataset
            - act_fn : Activation function used throughout the encoder network
        """
        super(CEncoder, self).__init__()
        c_hid = base_channel_size
        if dataset == 'omniglot':
            self.net = nn.Sequential(
                nn.Conv2d(num_input_channels, c_hid, kernel_size=3,
                          padding=1, stride=2),  # 28x28 => 16x16
                act_fn(),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.Conv2d(c_hid, 2*c_hid, kernel_size=3,
                          padding=1, stride=2),  # 16x16 => 8x8
                act_fn(),
                nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3,
                          padding=1, stride=2),  # 8x8 => 4x4
                act_fn(),
                nn.Flatten(),  # Image grid to single feature vector
            )
            self.h1 = nn.Linear(2*16*c_hid, latent_dim)
            self.h2 = nn.Linear(2*16*c_hid, latent_dim)

        elif dataset == 'mini_imagenet':
            self.net = nn.Sequential(
                nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_hid),
                act_fn(),
                nn.MaxPool2d(2),  # 28 x 28, # 42 x 42

                # nn.ZeroPad2d(conv_padding),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_hid),
                act_fn(),
                nn.MaxPool2d(2),  # 9x9 # 21 x 21

                # nn.ZeroPad2d(conv_padding),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_hid),
                act_fn(),
                nn.MaxPool2d(2),  # 3x3 # 10 x 10

                # nn.ZeroPad2d(conv_padding),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_hid),
                act_fn(),
                nn.MaxPool2d(2),  # 1x1 # 5 x 5
                nn.Flatten()
            )
            self.h1 = nn.Linear(c_hid*25, latent_dim)  # for maxpool(2)
            self.h2 = nn.Linear(c_hid*25, latent_dim)

    def forward(self, x):
        x = self.net(x)
        mu = self.h1(x)
        log_var = self.h2(x)
        return mu, log_var


class CDecoder(nn.Module):
    """ Convolutional Decoder for reconstructing an image using a latent variable z as input. """

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 dataset: str,
                 act_fn: object = nn.ReLU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct.
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers use a 2x of it.
            - latent_dim : Dimensionality of latent representation z + Dimensionality of one-hot encoded label 
            - act_fn : Activation function used throughout the decoder network
        """
        super(CDecoder, self).__init__()
        c_hid = base_channel_size
        self.dataset = dataset
        if self.dataset == 'omniglot':
            self.linear = nn.Sequential(
                nn.Linear(latent_dim, 2*16*c_hid),
                act_fn()
            )
            self.net = nn.Sequential(
                nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3,
                                   padding=1, stride=2),  # 4x4 => 8x8
                act_fn(),
                nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3,
                                   output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
                act_fn(),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3,
                                   output_padding=1, padding=1, stride=2),  # 16x16 => 32x32
                nn.Sigmoid()  # The input image is scaled between 0 and 1, hence the output has to be bounded as well
            )

        elif self.dataset == 'mini_imagenet':
            self.linear = nn.Sequential(
                nn.Linear(latent_dim, 25*c_hid),
                act_fn()
            )
            self.net = nn.Sequential(
                # nn.UpsamplingNearest2d(size=(5, 5)),
                # nn.Conv2d(in_channels=latent_dim, out_channels=c_hid, kernel_size=3, padding='same'),
                # nn.BatchNorm2d(c_hid),
                # act_fn(),

                nn.UpsamplingNearest2d(size=(10, 10)),
                nn.Conv2d(in_channels=c_hid, out_channels=c_hid,
                          kernel_size=3, padding='same'),
                nn.BatchNorm2d(c_hid),
                act_fn(),

                nn.UpsamplingNearest2d(size=(21, 21)),
                nn.Conv2d(in_channels=c_hid, out_channels=c_hid,
                          kernel_size=3, padding='same'),
                nn.BatchNorm2d(c_hid),
                act_fn(),

                nn.UpsamplingNearest2d(size=(42, 42)),
                nn.Conv2d(in_channels=c_hid, out_channels=c_hid,
                          kernel_size=3, padding='same'),
                nn.BatchNorm2d(c_hid),
                act_fn(),

                nn.UpsamplingNearest2d(size=(84, 84)),
                nn.Conv2d(in_channels=c_hid, out_channels=num_input_channels,
                          kernel_size=3, padding='same'),
                nn.BatchNorm2d(num_input_channels),
                nn.Sigmoid()
            )

    def forward(self, x):
        if self.dataset == 'omniglot':
            x = self.linear(x)
            x = x.reshape(x.shape[0], -1, 4, 4)
        elif self.dataset == 'mini_imagenet':
            #x = x.unsqueeze(-1).unsqueeze(-1)
            x = self.linear(x)
            x = x.reshape(x.shape[0], -1, 5, 5)
        x = self.net(x)
        return x


class CVAE(nn.Module):
    """ Module for a Convolutional VAE: Convolutional Encoder + Decoder. The decoder uses the 
    latent-space vector z drawn from the gaussian distribution and a one-hot encoded vector of 
    the class label to reconstruct the input. """

    def __init__(self, in_channels, y_shape, base_channels, latent_dim=64):
        super(CVAE, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.dec_latent_dim = self.latent_dim + y_shape

        self.encoder = CEncoder(num_input_channels=self.in_channels,
                                base_channel_size=self.base_channels, latent_dim=self.latent_dim)

        self.decoder = CDecoder(num_input_channels=self.in_channels,
                                base_channel_size=self.base_channels, latent_dim=self.dec_latent_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, y):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x = self.decoder(torch.cat([z, y], dim=1))
        return x, mu, log_var


class Classifier_VAE(nn.Module):
    """ Module for a Convolutional-VAE: Convolutional Encoder + Linear Classifier that 
    transforms an input image into latent-space gaussian distribution, and uses z_c drawn 
    from this distribution to produce logits for classification. """

    def __init__(self, in_channels, base_channels, latent_dim, n_ways, dataset, reparametrize=True, act_fn: object = nn.ReLU):
        super(Classifier_VAE, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.classes = n_ways
        self.reparameterize = reparametrize

        self.encoder = CEncoder(num_input_channels=self.in_channels,
                                base_channel_size=self.base_channels, latent_dim=self.latent_dim, dataset=dataset)
        
        self.classifier = nn.Sequential(
        nn.Linear(self.latent_dim, self.latent_dim//2), act_fn(),
        nn.Linear(self.latent_dim//2, self.classes))        

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, log_var = self.encoder(x)
        if self.reparametrize:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        logits = self.classifier(z)
        return logits, mu, log_var


class CCVAE(nn.Module):
    """ Module for a Conditional-Convolutional VAE: Classifier-VAE + Convolutional Encoder-Decoder. 
    The Conv. Encoder-Decoder is conditioned on the z_l drawn from the class-latent gaussian distribution 
    for reconstructing the input image. """

    def __init__(self, in_channels, base_channels, n_ways, dataset, latent_dim_l=64, latent_dim_s=64):
        super(CCVAE, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.dataset = dataset
        self.latent_dim_l = latent_dim_l
        self.latent_dim_s = latent_dim_s
        self.classes = n_ways

        self.encoder = CEncoder(num_input_channels=self.in_channels,
                                base_channel_size=self.base_channels, latent_dim=self.latent_dim_s, dataset=self.dataset)

        self.decoder = CDecoder(num_input_channels=self.in_channels,
                                base_channel_size=self.base_channels, latent_dim=(self.latent_dim_s + self.latent_dim_l), dataset=self.dataset)

        self.classifier_vae = Classifier_VAE(
            self.in_channels, self.base_channels, self.latent_dim_l, self.classes, dataset)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        logits, mu_l, log_var_l = self.classifier_vae(x)
        mu_s, log_var_s = self.encoder(x)
        z_s = self.reparameterize(mu_s, log_var_s)
        x = self.decoder(torch.cat([z_s, mu_l], dim=1))
        return x, logits, mu_l, log_var_l, mu_s, log_var_s


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, remove_linear=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if remove_linear:
            self.fc = None
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.fc is None:
            if feature:
                return x, None
            else:
                return x
        if feature:
            x1 = self.fc(x)
            return x, x1
        x = self.fc(x)
        return x
