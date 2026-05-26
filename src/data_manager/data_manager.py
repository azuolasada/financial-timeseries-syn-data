from src.data_manager.data_loader import DataLoader
from src.data_manager.data_processor import DataProcessor
from src.config_schema import DataConfig


class DataManager:
    def __init__(self, config: DataConfig):
        self.config = config
        self.loader = DataLoader(config=config)
        self.processor = DataProcessor(config=config)

        self.data = None
        self.scalers = None

    def get_data(self) -> None:
        raw_data = self.loader.load()
        self.data = self.processor.process(data=raw_data)
        self.scalers = self.processor.scalers