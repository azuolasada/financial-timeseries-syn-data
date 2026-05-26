from pathlib import Path
import ccxt
from datetime import datetime
import pandas as pd


class DataExtractor:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.time_format = '%Y-%m-%d'

    def extract_data(self, file_names: list[Path]) -> None:
        for file_name in file_names:
            symbol_data = self._extract_single_symbol(file_name=file_name)
            self._save_data(data=symbol_data, file_name=file_name)

    def _extract_single_symbol(self, file_name: Path) -> list[list[float]]:
        time_format = self.time_format
        name_base = file_name.stem
        parts = name_base.split('_')

        symbol = f"{parts[1]}/USDT"
        timeframe = parts[2]
        start_dt = datetime.strptime(parts[3], time_format)
        end_dt = datetime.strptime(parts[4], time_format)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        print(f"INFO: starting to extract data: {file_name}")

        result = []
        last_point = start_ms
        timeframe_ms = self.exchange.parse_timeframe(timeframe) * 1000
        while last_point < end_ms:
            batch = self.exchange.fetch_ohlcv(symbol=symbol,
                                              timeframe=timeframe,
                                              since=last_point,
                                              limit=1000)
            if not batch:
                break

            # Keep only candles strictly before end_ms
            clipped = [row for row in batch if row[0] < end_ms]
            result.extend(clipped)

            prev_point = last_point
            last_candle_start = batch[-1][0]
            last_point = last_candle_start + timeframe_ms

            if last_point <= prev_point:
                break

        print(f"INFO: extracted {len(result)} candles: {file_name}")

        return result

    @staticmethod
    def _save_data(data: list[list[float]], file_name: Path) -> None:
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df[['timestamp', 'open']].to_parquet(file_name, index=False)