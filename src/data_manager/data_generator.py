from pathlib import Path
from typing import NamedTuple, Optional
import pandas as pd
import re

from src.generation.stationary_bootstrap import StationaryBootstrapGen
from src.generation.quantgan import QuantGAN
from src.generation.tc_vae import TC_VAE

GENERATORS = {
    "SB": StationaryBootstrapGen,
    "QGAN": QuantGAN,
    "TCVAE": TC_VAE
}


class SyntheticFileInfo(NamedTuple):
    path: Path
    data_type: str
    real_symbol: str
    model_code: str
    path_idx: str
    timeframe: str
    start_date: str
    end_date: str


class DataGenerator:
    def __init__(self):
        # Synthetic symbol format: {real_symbol}_{model}{path_index}
        self.syn_symbol_pattern = re.compile(r"([A-Z]+)_([A-Z]+)(\d+)")

    def generate(self, missing_syn: list[Path]) -> None:
        groups = self._group_missing_files(missing_syn)

        for group_key, files in groups.items():
            self._process_group(group_key, files)

    def _group_missing_files(self, paths: list[Path]) -> dict[tuple, list[SyntheticFileInfo]]:
        groups = {}
        for path in paths:
            info = self._parse_synthetic_path(path)
            if not info:
                continue

            key = (info.real_symbol, info.model_code, info.timeframe, info.start_date, info.end_date)
            if key not in groups:
                groups[key] = []
            groups[key].append(info)
        return groups

    @staticmethod
    def _parse_synthetic_path(path: Path) -> Optional[SyntheticFileInfo]:
        # Filename format: {data_type}_{symbol}_{timeframe}_{start}_{end}.parquet
        # Example: train_BTC_SB0_1d_2020-01-01_2024-04-01.parquet
        parts = path.stem.split('_')

        data_type = parts[0]
        real_symbol = parts[1]
        model_with_idx = parts[2]
        timeframe = parts[3]
        start_date = parts[4]
        end_date = parts[5]

        # Extract model and index from model_with_idx
        match = re.match(r"([A-Z]+)(\d+)", model_with_idx)
        if not match:
            print(f"WARNING: Could not parse model/index from '{model_with_idx}' in {path.name}")
            return None

        model_code, path_idx = match.groups()

        return SyntheticFileInfo(
            path=path,
            data_type=data_type,
            real_symbol=real_symbol,
            model_code=model_code,
            path_idx=path_idx,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

    def _process_group(self, group_key: tuple, files: list[SyntheticFileInfo]) -> None:
        real_symbol, model_code, timeframe, start_date, end_date = group_key

        if model_code not in GENERATORS:
            print(f"WARNING: No generator found for model code {model_code}")
            return
        
        real_train_file = Path(f"data/train_{real_symbol}_{timeframe}_{start_date}_{end_date}.parquet")
        if not real_train_file or not real_train_file.exists():
            print(f"ERROR: Real training data for {real_symbol} missing. Cannot train {model_code}")
            return

        print(f"INFO: Training {model_code} for {real_symbol} using {real_train_file.name}")
        train_df = pd.read_parquet(real_train_file)

        gen_class = GENERATORS[model_code]
        generator = gen_class()
        generator.train(train_df)

        for file_info in files:
            self._generate_file(generator, file_info)

    @staticmethod
    def _generate_file(generator, file_info: SyntheticFileInfo) -> None:
        print(f"INFO: Generating synthetic data for {file_info.path.name}")

        syn_data = generator.generate()
        syn_data.to_parquet(file_info.path, index=False)
