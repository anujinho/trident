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


class GaussianParametrizer(nn.Module):
    """ Linear mapper from Image features to mean and log-variance parameters of latent-space gaussian distribution. """

    def __init__(self,
                 latent_dim: int,
                 feature_dim: int,
                 args,
                 act_fn: object = nn.ReLU):
        """
        Inputs:
            - latent_dim : Dimensionality of latent representation z
            - dataset: name of the dataset
            - feature_dim: dimensionality of the input feature
            - act_fn : Activation function used throughout the network (if at all)
        """
        super(GaussianParametrizer, self).__init__()

        self.args = args

        if (args.dataset == 'omniglot') or (args.dataset == 'cifarfs'):
            self.h1 = nn.Linear(feature_dim, latent_dim)
            self.h2 = nn.Linear(feature_dim, latent_dim)
        elif (args.dataset == 'miniimagenet') or (args.dataset == 'tiered'):
            self.h1 = nn.Linear(feature_dim, latent_dim)
            self.h2 = nn.Linear(feature_dim, latent_dim)

    def forward(self, x):
        mu = self.h1(x)
        log_var = self.h2(x)
        return mu, log_var


class CEncoder(nn.Module):
    """ Convolutional Encoder to transform an input image into its flattened feature embedding. """

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 dataset: str,
                 args,
                 act_fn: object = nn.ReLU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers use 2x of it.
            - dataset: name of the dataset
            - act_fn : Activation function used throughout the encoder network
        """
        super(CEncoder, self).__init__()
        c_hid = base_channel_size
        self.args = args
        if dataset == 'omniglot':
            self.net = nn.Sequential(
                nn.Conv2d(num_input_channels, c_hid,
                          kernel_size=3, padding=1, stride=(2, 2)),
                nn.BatchNorm2d(c_hid),
                act_fn(),  # 14

                nn.Conv2d(c_hid, c_hid, kernel_size=3,
                          padding=1, stride=(2, 2)),
                nn.BatchNorm2d(c_hid),
                act_fn(),  # 7

                nn.Conv2d(c_hid, c_hid, kernel_size=3,
                          padding=1, stride=(2, 2)),
                nn.BatchNorm2d(c_hid),
                act_fn(),  # 4

                nn.Conv2d(c_hid, c_hid, kernel_size=3,
                          padding=1, stride=(2, 2)),
                nn.BatchNorm2d(c_hid),
                act_fn(),  # 2

                nn.Flatten()
            )

        elif (dataset == 'miniimagenet') or (dataset == 'cifarfs') or (dataset == 'tiered'):
            self.net = nn.Sequential(
                nn.Conv2d(num_input_channels, c_hid,
                          kernel_size=3, padding=1),
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

    def forward(self, x):
        x = self.net(x)
        return x


class TADCEncoder(nn.Module):
    """ Convolutional Encoder to transform an input image into its task/episode aware feature embedding. """

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 dataset: str,
                 task_adapt_fn: str,
                 args,
                 act_fn: object = nn.ReLU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers use 2x of it.
            - dataset: name of the dataset
            - task_adapt_fn: EAEN (eaen) or Kernel Smoothing (gks)
            - args: dict of arguments
            - act_fn : Activation function used throughout the encoder network
        """

        super(TADCEncoder, self).__init__()
        c_hid = base_channel_size
        self.args = args
        self.task_adapt_fn = task_adapt_fn

        if dataset == 'omniglot':
            self.net = nn.Sequential(
                nn.Conv2d(num_input_channels, c_hid,
                          kernel_size=3, padding=1, stride=(2, 2)),
                nn.BatchNorm2d(c_hid),
                act_fn(),  # 14

                nn.Conv2d(c_hid, c_hid, kernel_size=3,
                          padding=1, stride=(2, 2)),
                nn.BatchNorm2d(c_hid),
                act_fn(),  # 7

                nn.Conv2d(c_hid, c_hid, kernel_size=3,
                          padding=1, stride=(2, 2)),
                nn.BatchNorm2d(c_hid),
                act_fn(),  # 4

                nn.Conv2d(c_hid, c_hid, kernel_size=3,
                          padding=1, stride=(2, 2)),
                nn.BatchNorm2d(c_hid),
                act_fn()  # 2
            )

        elif (dataset == 'miniimagenet') or (dataset == 'cifarfs') or (dataset == 'tiered'):
            self.net = nn.Sequential(
                nn.Conv2d(num_input_channels, c_hid,
                          kernel_size=3, padding=1),
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
                nn.MaxPool2d(2)  # 1x1 # 5 x 5
            )

        self.n = args.n_ways * (args.k_shots + args.q_shots)
        self.eaen = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(
                self.n, 1), stride=(1, 1), padding='valid', bias=False),
            # P
            act_fn(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(
                1, 1), stride=(1, 1), padding='valid', bias=False),
            # Z
            act_fn(),

            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(
                1, 1), stride=(1, 1), padding='valid', bias=False),
            # F
            act_fn(),
        )

    def forward(self, x, update: str):
        x = self.net(x)

        # Task aware embeddings

        if self.task_adapt_fn == 'eaen':
            G = x.permute(2, 3, 0, 1)
            G = G.reshape(G.shape[0] * G.shape[1],
                          G.shape[2], G.shape[3]).unsqueeze(dim=1)
            G = self.eaen(G)
            G = G.squeeze().transpose(0, 1).reshape(-1, x.shape[2], x.shape[3])
            if update == 'inner':
                x = x[:self.args.n_ways*self.args.k_shots] * G
            elif update == 'outer':
                x = x[self.args.n_ways*self.args.k_shots:] * G
            x = nn.Flatten()(x)

        elif self.task_adapt_fn == 'gks':
            G = x.reshape(self.n, -1)
            A = torch.cdist(G, G, p=2) ** 2
            A = A / A.var()  # Normalized adjacency matrix
            D = torch.diag(A.sum(dim=1).pow(-0.5))
            L = torch.matmul(torch.matmul(D, A), D)  # Laplacian Matrix
            I = torch.eye(self.n, self.n).to(self.args.device)
            P = torch.linalg.inv(I - self.args.alpha * L)  # Propagator Matrix
            x = torch.matmul(P, G)
            if update == 'inner':
                x = x[:self.args.n_ways*self.args.k_shots]
            elif update == 'outer':
                x = x[self.args.n_ways*self.args.k_shots:]

        return x


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
                nn.Linear(latent_dim, 4*c_hid),
                act_fn()
            )
            self.net = nn.Sequential(
                nn.UpsamplingNearest2d(size=(4, 4)),
                nn.Conv2d(in_channels=c_hid, out_channels=c_hid,
                          kernel_size=3, padding='same'),
                nn.BatchNorm2d(c_hid),
                act_fn(),

                nn.UpsamplingNearest2d(size=(7, 7)),
                nn.Conv2d(in_channels=c_hid, out_channels=c_hid,
                          kernel_size=3, padding='same'),
                nn.BatchNorm2d(c_hid),
                act_fn(),

                nn.UpsamplingNearest2d(size=(14, 14)),
                nn.Conv2d(in_channels=c_hid, out_channels=c_hid,
                          kernel_size=3, padding='same'),
                nn.BatchNorm2d(c_hid),
                act_fn(),

                nn.UpsamplingNearest2d(size=(28, 28)),
                nn.Conv2d(in_channels=c_hid, out_channels=num_input_channels,
                          kernel_size=3, padding='same'),
                nn.BatchNorm2d(num_input_channels),
                nn.Sigmoid()
            )

        elif (self.dataset == 'miniimagenet') or (self.dataset == 'cifarfs') or (self.dataset == 'tiered'):
            if (self.dataset == 'miniimagenet') or (self.dataset == 'tiered'):
                self.linear = nn.Sequential(
                    nn.Linear(latent_dim, 25*c_hid),
                    act_fn()
                )
                a1 = 10
                a2 = 21
                a3 = 42
                a4 = 84
            elif self.dataset == 'cifarfs':
                self.linear = nn.Sequential(
                    nn.Linear(latent_dim, 4*c_hid),
                    act_fn()
                )
                a1 = 4
                a2 = 8
                a3 = 16
                a4 = 32

            self.net = nn.Sequential(

                nn.UpsamplingNearest2d(size=(a1, a1)),
                nn.Conv2d(in_channels=c_hid, out_channels=c_hid,
                          kernel_size=3, padding='same'),
                nn.BatchNorm2d(c_hid),
                act_fn(),

                nn.UpsamplingNearest2d(size=(a2, a2)),
                nn.Conv2d(in_channels=c_hid, out_channels=c_hid,
                          kernel_size=3, padding='same'),
                nn.BatchNorm2d(c_hid),
                act_fn(),

                nn.UpsamplingNearest2d(size=(a3, a3)),
                nn.Conv2d(in_channels=c_hid, out_channels=c_hid,
                          kernel_size=3, padding='same'),
                nn.BatchNorm2d(c_hid),
                act_fn(),

                nn.UpsamplingNearest2d(size=(a4, a4)),
                nn.Conv2d(in_channels=c_hid, out_channels=num_input_channels,
                          kernel_size=3, padding='same'),
                nn.BatchNorm2d(num_input_channels),
                nn.Sigmoid()
            )

    def forward(self, x):
        if (self.dataset == 'omniglot') or (self.dataset == 'cifarfs'):
            x = self.linear(x)
            x = x.reshape(x.shape[0], -1, 2, 2)
        elif (self.dataset == 'miniimagenet') or (self.dataset == 'tiered'):
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
    transforms an input image into latent-space gaussian distribution, and uses z_l drawn 
    from this distribution to produce logits for classification. """

    def __init__(self, in_channels, base_channels, latent_dim_l, latent_dim_s, n_ways, dataset, task_adapt, task_adapt_fn, args, act_fn: object = nn.ReLU):
        super(Classifier_VAE, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim_l = latent_dim_l
        self.latent_dim_s = latent_dim_s
        self.classes = n_ways
        self.task_adapt = task_adapt
        self.task_adapt_fn = task_adapt_fn

        fcoeff = 25 if (dataset == 'miniimagenet') or (
            dataset == 'tiered') else 4
        fsize = fcoeff*self.base_channels

        if self.task_adapt:
            self.encoder = TADCEncoder(num_input_channels=self.in_channels,
                                       base_channel_size=self.base_channels, dataset=dataset, task_adapt_fn=self.task_adapt_fn, args=args)
        else:
            self.encoder = CEncoder(num_input_channels=self.in_channels,
                                    base_channel_size=self.base_channels, dataset=dataset, args=args)

        self.gaussian_parametrizer = GaussianParametrizer(
            latent_dim=self.latent_dim_l, feature_dim=(fsize + self.latent_dim_s), args=args)

        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim_l, self.latent_dim_l//2), act_fn(),
            nn.Linear(self.latent_dim_l//2, self.classes))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, z_s, update):
        if self.task_adapt:
            x = self.encoder(x, update)
        else:
            x = self.encoder(x)
        mu_l, log_var_l = self.self.gaussian_parametrizer(
            torch.cat([x, z_s]), dim=1)
        z_l = self.reparameterize(mu_l, log_var_l)
        logits = self.classifier(z_l)
        return logits, mu_l, log_var_l, z_l


class CCVAE(nn.Module):
    """ Module for a Conditional-Convolutional VAE: Classifier-VAE + Convolutional Encoder-Decoder. 
    The Conv. Encoder-Decoder is conditioned on the z_l drawn from the class-latent gaussian distribution 
    for reconstructing the input image. """

    def __init__(self, in_channels, base_channels, n_ways, dataset, task_adapt, task_adapt_fn, args, latent_dim_l=64, latent_dim_s=64):
        super(CCVAE, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.dataset = dataset
        self.latent_dim_l = latent_dim_l
        self.latent_dim_s = latent_dim_s
        self.classes = n_ways
        self.task_adapt = task_adapt
        self.task_adapt_fn = task_adapt_fn
        self.args = args

        fcoeff = 25 if (dataset == 'miniimagenet') or (
            dataset == 'tiered') else 4

        self.encoder = CEncoder(num_input_channels=self.in_channels,
                                base_channel_size=self.base_channels, dataset=self.dataset, args=args)

        self.decoder = CDecoder(num_input_channels=self.in_channels,
                                base_channel_size=self.base_channels, latent_dim=(self.latent_dim_s + self.latent_dim_l), dataset=self.dataset)

        self.classifier_vae = Classifier_VAE(
            in_channels=self.in_channels, base_channels=self.base_channels, latent_dim_l=self.latent_dim_l, latent_dim_s=self.latent_dim_s, n_ways=self.classes, dataset=dataset, task_adapt=task_adapt, task_adapt_fn=task_adapt_fn, args=self.args)

        self.gaussian_parametrizer = GaussianParametrizer(
            latent_dim=self.latent_dim_s, feature_dim=fcoeff, args=args)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, update):
        if self.task_adapt & (update == 'inner'):
            xs = x[:self.args.n_ways*self.args.k_shots]
        elif self.task_adapt & (update == 'outer'):
            xs = x[self.args.n_ways*self.args.k_shots:]
        else:
            xs = x
        
        xs = self.encoder(xs)
        mu_s, log_var_s = self.gaussian_parametrizer(xs)
        z_s = self.reparameterize(mu_s, log_var_s)
        del xs

        logits, mu_l, log_var_l, z_l = self.classifier_vae(x, z_s, update)
        x = self.decoder(torch.cat([z_s, z_l], dim=1))

        return x, logits, mu_l, log_var_l, mu_s, log_var_s, z_s


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        drop_rate=0.0,
        drop_block=False,
        block_size=1,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(
                    1.0 - self.drop_rate / 40000 * self.num_batches_tracked,
                    1.0 - self.drop_rate
                )
                gamma = (
                    (1 - keep_rate)
                    / self.block_size**2 * feat_size**2
                    / (feat_size - self.block_size + 1)**2
                )
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(
                    out,
                    p=self.drop_rate,
                    training=self.training,
                    inplace=True,
                )
        return out


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()
        self.block_size = block_size

    def forward(self, x, gamma):

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = torch.distributions.Bernoulli(gamma)
            mask = bernoulli.sample((
                batch_size,
                channels,
                height - (self.block_size - 1),
                width - (self.block_size - 1),
            )).to(x.device)
            block_mask = self._compute_block_mask(mask)
            countM = (
                block_mask.size(0)
                * block_mask.size(1)
                * block_mask.size(2)
                * block_mask.size(3)
            )
            count_ones = block_mask.sum()
            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size-1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        non_zero_idxs = mask.nonzero(as_tuple=False)
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(
                    self.block_size,
                    self.block_size).reshape(-1),
                torch.arange(self.block_size).repeat(self.block_size),
            ]
        ).t()
        offsets = torch.cat(
            (torch.zeros(self.block_size**2, 2).long(), offsets.long()),
            dim=1,
        ).to(mask.device)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(
                mask,
                (left_padding, right_padding, left_padding, right_padding)
            )
            padded_mask[
                block_idxs[:, 0],
                block_idxs[:, 1],
                block_idxs[:, 2],
                block_idxs[:, 3]] = 1.0
        else:
            padded_mask = F.pad(
                mask,
                (left_padding, right_padding, left_padding, right_padding)
            )

        block_mask = 1 - padded_mask
        return block_mask


class ResNet12Backbone(nn.Module):

    def __init__(
        self,
        args,
        avg_pool=True,  # Set to False for 16000-dim embeddings
        wider=True,  # True mimics MetaOptNet, False mimics TADAM
        embedding_dropout=0.0,  # dropout for embedding
        dropblock_dropout=0.1,  # dropout for residual layers
        dropblock_size=5,
        channels=3,
    ):
        super(ResNet12Backbone, self).__init__()
        self.args = args
        self.inplanes = channels
        block = BasicBlock
        if wider:
            num_filters = [64, 160, 320, 640]
        else:
            num_filters = [64, 128, 256, 512]

        self.layer1 = self._make_layer(
            block,
            num_filters[0],
            stride=2,
            dropblock_dropout=dropblock_dropout,
        )
        self.layer2 = self._make_layer(
            block,
            num_filters[1],
            stride=2,
            dropblock_dropout=dropblock_dropout,
        )
        self.layer3 = self._make_layer(
            block,
            num_filters[2],
            stride=2,
            dropblock_dropout=dropblock_dropout,
            drop_block=True,
            block_size=dropblock_size,
        )
        self.layer4 = self._make_layer(
            block,
            num_filters[3],
            stride=2,
            dropblock_dropout=dropblock_dropout,
            drop_block=True,
            block_size=dropblock_size,
        )
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        else:
            self.avgpool = Lambda(lambda x: x)
        self.flatten = Flatten()
        self.embedding_dropout = embedding_dropout
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=self.embedding_dropout, inplace=False)
        self.dropblock_dropout = dropblock_dropout

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='leaky_relu',
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        planes,
        stride=1,
        dropblock_dropout=0.0,
        drop_block=False,
        block_size=1,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(
            self.inplanes,
            planes,
            stride,
            downsample,
            dropblock_dropout,
            drop_block,
            block_size)
        )
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.args.pretrained[2] == 640:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.dropout(x)
        elif self.args.pretrained[2] == 16000:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)

        return x


def conv3x3wrn(in_planes, out_planes, stride=1):
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True,
    )


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform(m.weight, gain=2**0.5)
        torch.nn.init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.constant(m.weight, 1)
        torch.nn.init.constant(m.bias, 0)


class wide_basic(torch.nn.Module):

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(in_planes)
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=True
                ),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(torch.nn.Module):

    def __init__(self, depth, widen_factor, dropout_rate):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3wrn(3, nStages[0])
        self.layer1 = self._wide_layer(
            wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(
            wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(
            wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(nStages[3], momentum=0.9)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 21)
        out = out.view(out.size(0), -1)
        return out


class WRN28Backbone(WideResNet):

    def __init__(self, dropout=0.0):
        super(WRN28Backbone, self).__init__(
            depth=28,
            widen_factor=10,
            dropout_rate=dropout,
        )
