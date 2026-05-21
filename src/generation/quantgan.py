import pandas as pd
import numpy as np
from tqdm import tqdm

from scipy.stats import kurtosis
from scipy.optimize import fmin
from scipy.special import lambertw

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class QuantGAN:
    """
    Implementation base used from https://github.com/KseniaKingsep/quantgan/tree/main
    """
    def __init__(self,
                 batch_size: int = 30,
                 num_epochs: int = 200,
                 learning_rate: float = 0.0002,
                 nz: int = 3,
                 clip_value: float = 0.01,
                 seq_len: int = 127,
                 random_state: int = 50):
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.nz = nz
        self.clip_value = clip_value
        self.seq_len = seq_len

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.data = None
        self.max_val = None
        self.params = None

        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        np.random.seed(random_state)

    def train(self, data: pd.DataFrame):
        self.data = data.copy()

        prepped_data = self.prep_data(data=data)

        dataset = CryptoDataset(prepped_data, self.seq_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        netG = Generator(self.nz, 1).to(self.device)
        netD = Discriminator(1, 1).to(self.device)
        optD = optim.RMSprop(netD.parameters(), lr=self.learning_rate)
        optG = optim.RMSprop(netG.parameters(), lr=self.learning_rate)

        t = tqdm(range(self.num_epochs))
        for epoch in t:
            for i, data in enumerate(dataloader, 0):

                netD.zero_grad()
                real = data.to(self.device)
                batch_size, seq_len = real.size(0), real.size(1)
                noise = torch.randn(batch_size, seq_len, self.nz, device=self.device)
                fake = netG(noise).detach()

                lossD = -torch.mean(netD(real)) + torch.mean(netD(fake))
                lossD.backward()
                optD.step()

                for p in netD.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)

                noise = torch.randn(batch_size, seq_len, self.nz, device=self.device)
                netG.zero_grad()
                lossG = -torch.mean(netD(netG(noise)))
                lossG.backward()
                optG.step()
                # Report metrics
            t.set_description('Loss_D: %.8f Loss_G: %.8f' % (lossD.item(), lossG.item()))

        print(f"Training finished. Final Loss_D: {lossD.item():.8f}, Final Loss_G: {lossG.item():.8f}")
        self.model = netG

    def generate(self):
        data_len = len(self.data) - 1
        noise = torch.randn(1, data_len, self.nz, device=self.device)
        generated = self.model(noise).detach().cpu().reshape(data_len).numpy()

        log_returns = self.inverse(generated * self.max_val, self.params)

        initial_price = self.data.iloc[0]['open']
        price_path = initial_price * np.exp(np.cumsum(log_returns))
        full_price_path = np.insert(price_path, 0, initial_price)

        df = pd.DataFrame({
            'timestamp': self.data['timestamp'].values,
            'open': full_price_path
        })
        return df

    def prep_data(self, data: pd.DataFrame):
        data['log_ret'] = np.log(data['open'] / data['open'].shift(1))
        data.dropna(inplace=True)

        log_mean = np.mean(data['log_ret'])
        log_norm = data['log_ret'] - log_mean
        params = self.igmm(log_norm)
        processed = self.w_delta((log_norm - params[0]) / params[1], params[2])
        max_val = np.max(np.abs(processed))
        processed = processed / max_val

        self.max_val = max_val
        self.params = params

        return processed

    def igmm(self, z, eps=1e-6, max_iter=100):
        delta = self.delta_init(z)
        params = [np.median(z), np.std(z) * (1. - 2. * delta) ** 0.75, delta]
        for k in range(max_iter):
            params_old = params
            u = (z - params[0]) / params[1]
            params[2] = self.delta_gmm(u)
            x = self.w_params(z, params)
            params[0], params[1] = np.mean(x), np.std(x)

            if np.linalg.norm(np.array(params) - np.array(params_old)) < eps:
                break
            if k == max_iter - 1:
                raise "Solution not found"

        return params

    @staticmethod
    def delta_init(z):
        k = kurtosis(z, fisher=False, bias=False)
        if k < 166. / 62.:
            return 0.01
        return np.clip(1. / 66 * (np.sqrt(66 * k - 162.) - 6.), 0.01, 0.48)

    def delta_gmm(self, z):
        delta = self.delta_init(z)

        def iteration(q):
            u = self.w_delta(z, np.exp(q))
            if not np.all(np.isfinite(u)):
                return 0.
            k = kurtosis(u, fisher=True, bias=False) ** 2
            if not np.isfinite(k) or k > 1e10:
                return 1e10
            return k

        res = fmin(iteration, np.log(delta), disp=0)
        return np.around(np.exp(res[-1]), 6)

    @staticmethod
    def w_delta(z, delta):
        return np.sign(z) * np.sqrt(np.real(lambertw(delta * z ** 2)) / delta)

    def w_params(self, z, params):
        return params[0] + params[1] * self.w_delta((z - params[0]) / params[1], params[2])

    @staticmethod
    def inverse(z, params):
        return params[0] + params[1] * (z * np.exp(z * z * (params[2] * 0.5)))

class CryptoDataset(Dataset):
    def __init__(self, data, window):
        self.data = data
        self.window = window

    def __getitem__(self, index):
        x = np.expand_dims(self.data[index:index+self.window], -1)
        return torch.from_numpy(x).float()

    def __len__(self):
        return len(self.data) - self.window


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, kernel_size, dilation):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_hidden, kernel_size, stride=1, dilation=dilation, padding='same')

        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(n_hidden, n_outputs, kernel_size, stride=1, dilation=dilation, padding='same')
        self.relu2 = nn.PReLU()

        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCN(nn.Module):
    def __init__(self, input_size, output_size, n_hidden=80):
        super(TCN, self).__init__()
        layers = []
        dilation = 1
        for i in range(7):
            num_inputs = input_size if i == 0 else n_hidden
            kernel_size = 2 if i > 0 else 1
            if i == 0:
                dilation = 1
            elif i == 1:
                dilation = 1
            else:
                dilation = dilation * 2

            layers += [TemporalBlock(num_inputs, n_hidden, n_hidden, kernel_size, dilation)]
        self.conv = nn.Conv1d(n_hidden, output_size, 1)
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.net(x.transpose(1, 2))
        return self.conv(y1).transpose(1, 2)


class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.net = TCN(input_size, output_size)

    def forward(self, x):
        return torch.tanh(self.net(x))


class Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        self.net = TCN(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.net(x))

