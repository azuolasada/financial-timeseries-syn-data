import os
import numpy as np
from typing import Literal
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.prediction.pred_models import PredModel
from src.config_schema import PredictConfig


class ModelTrainer:
    def __init__(self,
                 model_name: Literal['LSTM', 'BD_LSTM', 'TCN'],
                 pred_cfg: PredictConfig,
                 n_steps_in: int = 6,
                 n_steps_out: int = 5):

        self.cfg = pred_cfg

        self.model = PredModel(model_name,
                               n_steps_in,
                               n_steps_out,
                               self.cfg.learning_rate)

        self.checkpoint_path = "best_model.keras"
        self.callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.cfg.early_stop_patience,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                self.checkpoint_path,
                monitor="val_loss",
                save_best_only=True
            )
        ]
        print("GPU device:", tf.test.gpu_device_name())

    def train(self,
              x_train: np.ndarray,
              y_train: np.ndarray,
              x_val: np.ndarray,
              y_val: np.ndarray,
              sample_weight: np.ndarray = None):
        print(f"Starting training for {self.model.model_name}...")
        # Reshape for LSTM/Conv1D: [samples, time_steps, features]
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))

        history = self.model.model.fit(
            x=x_train,
            y=y_train,
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            validation_data=(x_val, y_val),
            callbacks = self.callbacks,
            sample_weight=sample_weight,
            verbose=1
        )
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

        return history

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        x_test_reshaped = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        results = self.model.model.evaluate(x_test_reshaped, y_test, verbose=0)
        y_pred = self.model.model.predict(x_test_reshaped, verbose=0)
        print(f"Test results: {results}")
        return results, y_pred