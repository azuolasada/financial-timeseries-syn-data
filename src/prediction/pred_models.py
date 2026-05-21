from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tcn import TCN

class PredModel:
    """
    LSTM and Bidirectional LSTM models implementation used from: https://github.com/sydney-machine-learning/deeplearning-crypto/tree/main
    TCN implementation from: https://github.com/NathanPortelli/ARI5123-Intelligent-Algorithmic-Trading/tree/main
    """
    MODEL_BUILDERS = {
        'LSTM':    '_build_lstm',
        'BD_LSTM': '_build_bd_lstm',
        'TCN':     '_build_tcn',
    }

    def __init__(self,
                 model_name: str,
                 n_steps_in: int,
                 n_steps_out: int,
                 learning_rate: float) -> None:
        self.model_name = model_name
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.learning_rate = learning_rate
        self.model = self._create_model()

    def _create_model(self) -> Sequential:
        builder_name = self.MODEL_BUILDERS.get(self.model_name)

        if builder_name is None:
            raise ValueError(
                f"Unknown model_name '{self.model_name}'. "
                f"Choose from: {list(self.MODEL_BUILDERS)}"
            )

        model = Sequential()
        builder = getattr(self, builder_name)
        model = builder(model)

        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='mse',
                      metrics=["mse", "mae", "mape"])

        print(model.summary())
        return model

    def _build_lstm(self, model: Sequential) -> Sequential:
        model.add(LSTM(100, activation='sigmoid', return_sequences=True,
                       input_shape=(self.n_steps_in, 1)))
        model.add(LSTM(100, activation='sigmoid'))
        model.add(Dense(self.n_steps_out))
        return model

    def _build_bd_lstm(self, model: Sequential) -> Sequential:
        model.add(Bidirectional(LSTM(50, activation='sigmoid'),
                                input_shape=(self.n_steps_in, 1)))
        model.add(Dense(self.n_steps_out))
        return model

    def _build_tcn(self, model: Sequential) -> Sequential:
        model.add(TCN(
            nb_filters=64,
            kernel_size=3,
            dilations=[1, 2, 4, 8],
            return_sequences=True,
            input_shape=(self.n_steps_in, 1)
        ))
        model.add(Dropout(0.1))
        model.add(TCN(
            nb_filters=64,
            kernel_size=3,
            dilations=[1, 2, 4, 8, 16],
            return_sequences=False
        ))
        model.add(Dropout(0.1))
        model.add(Dense(self.n_steps_out))
        return model
