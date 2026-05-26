import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.config_schema import DataConfig


class DataProcessor:
    def __init__(self, config: DataConfig):
        self.n_steps_in = config.n_steps_in
        self.n_steps_out = config.n_steps_out

        self.feature_names = ['log_ret']
        self.feature_names.extend([f'log_ret_{i}' for i in range(1, self.n_steps_in)])

        self.target_names = [f'target_{i}' for i in range(1, self.n_steps_out + 1)]

        self.scalers = None

    def process(self, data: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
        processed_data = {}
        scalers = {}

        all_symbols = set()
        for key in data.keys():
            symbol = key.split('_')[0]
            all_symbols.add(symbol)

        for symbol in all_symbols:
            symbol_train_keys = [key for key in data.keys() if key.startswith(f"{symbol}_") and 'train' in key]
            
            prepped_train_dfs = []
            for key in symbol_train_keys:
                df = data[key]
                prepped_df = self._prep_data(df=df)
                prepped_train_dfs.append(prepped_df)

            combined_train_df = pd.concat(prepped_train_dfs)

            scaler_x = MinMaxScaler()
            scaler_y = MinMaxScaler()

            scaler_x.fit(combined_train_df[self.feature_names])
            scaler_y.fit(combined_train_df[self.target_names])

            scalers[f"{symbol}_x"] = scaler_x
            scalers[f"{symbol}_y"] = scaler_y

            # Transform and store all training sets for this symbol
            all_scaled_x = []
            all_scaled_y = []
            for key, prepped_df in zip(symbol_train_keys, prepped_train_dfs):
                scaled_x = scaler_x.transform(prepped_df[self.feature_names])
                scaled_y = scaler_y.transform(prepped_df[self.target_names])
                processed_data[f"{key}_x"] = scaled_x
                processed_data[f"{key}_y"] = scaled_y
                all_scaled_x.append(scaled_x)
                all_scaled_y.append(scaled_y)

            # 6. Store combined training data_analysis (to be used in main.py)
            processed_data[f"{symbol}_train_x"] = np.concatenate(all_scaled_x)
            processed_data[f"{symbol}_train_y"] = np.concatenate(all_scaled_y)

            # 7. Transform and store validation/test data_analysis for this symbol
            val_test_keys = [key for key in data.keys() 
                             if key.startswith(f"{symbol}_") and ('val' in key or 'test' in key)]
            
            for key in val_test_keys:
                df = data[key]
                prepped_df = self._prep_data(df=df)
                processed_data[f"{key}_x"] = scaler_x.transform(prepped_df[self.feature_names])
                processed_data[f"{key}_y"] = scaler_y.transform(prepped_df[self.target_names])

        self.scalers = scalers
        return processed_data

    def _prep_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create a copy to avoid modifying original dataframe
        df = df.copy()

        df['log_ret'] = np.log(df['open'] / df['open'].shift(1))

        # Add lagged log returns as features
        for i in range(1, self.n_steps_in):
            df[f'log_ret_{i}'] = df['log_ret'].shift(i)

        # Add future log returns as targets
        for i in range(1, self.n_steps_out + 1):
            df[f'target_{i}'] = df['log_ret'].shift(-i)

        return df[self.feature_names + self.target_names].dropna()