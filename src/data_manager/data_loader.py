from pathlib import Path
import pandas as pd

from src.config_schema import DataConfig
from src.data_manager.data_extractor import DataExtractor
from src.data_manager.data_generator import DataGenerator


class DataLoader:
    def __init__(self, config: DataConfig):
        self.symbols = config.symbols.copy()
        self.synthetic_paths = config.synthetic_paths
        self.syn_models = config.syn_models
        if self.synthetic_paths != 0 and len(self.syn_models) > 0:
            synthetic = [f"{symbol}_{model}{path}"
                         for symbol in self.symbols
                         for model in self.syn_models
                         for path in range(self.synthetic_paths)
                         ]
            self.symbols += synthetic
        self.timeframe = config.timeframe

        self.train_start_dt = config.train_start_dt
        self.train_end_dt = config.train_end_dt
        self.val_start_dt = config.val_start_dt
        self.val_end_dt = config.val_end_dt
        self.test_start_dt = config.test_start_dt
        self.test_end_dt = config.test_end_dt

        self.extractor = DataExtractor()
        self.generator = DataGenerator()

    def load(self) -> dict[str, pd.DataFrame]:
        # Get files that are needed to be loaded
        required_real, required_syn = self._get_required_files()
        missing_real = [file for file in required_real if not file.is_file()]
        missing_syn = [file for file in required_syn if not file.is_file()]

        if missing_real:
            self.extractor.extract_data(missing_real)
        
        if missing_syn:
            self.generator.generate(missing_syn)

        required_files = required_real + required_syn
        print(f"INFO: starting to load {len(required_files)} data_analysis files")
        all_data = {}
        for file in required_files:
            data = pd.read_parquet(file)
            data_type = file.stem.split('_')[0]
            symbol = file.stem.split('_')[1]
            syn = file.stem.split('_')[2]

            if syn == self.timeframe:
                key = f"{symbol}_{data_type}"
            else:
                key = f"{symbol}_{syn}_{data_type}"
            print(key)
            all_data[key] = data
        print("INFO: data_analysis loading completed")
        return all_data

    def _get_required_files(self) -> tuple[list[Path], list[Path]]:
        data_dir = Path("data_analysis")
        intervals = {
            "train": (self.train_start_dt, self.train_end_dt),
            "val": (self.val_start_dt, self.val_end_dt),
            "test": (self.test_start_dt, self.test_end_dt),
        }

        files_real = []
        files_syn = []

        for symbol in self.symbols:
            synthetic = any(c.isdigit() for c in symbol)
            target = files_syn if synthetic else files_real
            for data_type, (start, end) in intervals.items():
                if not (data_type != "train" and synthetic):
                    target.append(data_dir / f"{data_type}_{symbol}_{self.timeframe}_{start}_{end}.parquet")

        return files_real, files_syn