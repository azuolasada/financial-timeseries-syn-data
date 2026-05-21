from pydantic import BaseModel
from datetime import date

class DataConfig(BaseModel):
    symbols: list[str]
    synthetic_paths: int
    syn_models: list[str]
    timeframe: str
    train_start_dt: date
    train_end_dt: date
    val_start_dt: date
    val_end_dt: date
    test_start_dt: date
    test_end_dt: date
    n_steps_in: int
    n_steps_out: int

class PredictConfig(BaseModel):
    models: list[str]
    epochs: int
    learning_rate: float
    batch_size: int
    early_stop_patience: int

class ProjectConfig(BaseModel):
    data: DataConfig
    pred: PredictConfig