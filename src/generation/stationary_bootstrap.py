import pandas as pd
import numpy as np

from arch.bootstrap import StationaryBootstrap
from arch.bootstrap import optimal_block_length

class StationaryBootstrapGen:
    def __init__(self, block_size: int = None, random_state: int = 50):
        self.block_size = block_size
        self.random_state = random_state

        self.model = None
        self.data = None

    def train(self, data: pd.DataFrame) -> None:
        self.data = data.copy()

        data['log_ret'] = np.log(data['open'] / data['open'].shift(1))
        data.dropna(inplace=True)

        if not self.block_size:
            optimal_block = optimal_block_length(data['log_ret']).loc["log_ret", "stationary"]
            self.block_size = int(optimal_block)

        if self.block_size == 0:
            self.block_size = 7

        self.model = StationaryBootstrap(self.block_size,
                                         data['log_ret'].values,
                                         seed=self.random_state)

    def generate(self) -> pd.DataFrame:
        for gen_data, _ in self.model.bootstrap(1):
            boot_returns = gen_data[0].flatten()

        initial_price = self.data.iloc[0]['open']
        price_path = initial_price * np.exp(np.cumsum(boot_returns))
        full_price_path = np.insert(price_path, 0, initial_price)

        df = pd.DataFrame({
            'timestamp': self.data['timestamp'].values,
            'open': full_price_path
        })
        return df