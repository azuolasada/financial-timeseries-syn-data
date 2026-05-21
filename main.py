import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from src.config_schema import ProjectConfig
from src.data.data_manager import DataManager
from src.prediction.pred_trainer import ModelTrainer

# Read the config files
if not Path("config.yaml").is_file() or not Path("config_syn.yaml").is_file():
    raise FileNotFoundError("config.yaml or config_syn.yaml not found")

with open("config.yaml") as f:
    raw_cfg = yaml.safe_load(f)

with open("config_syn.yaml") as f:
    raw_cfg_syn = yaml.safe_load(f)

# Validate configuration
try:
    cfg = ProjectConfig(**raw_cfg)
    cfg_syn = ProjectConfig(**raw_cfg_syn)
except Exception as e:
    print(f"Configuration error: {e}")
    exit(1)

# ===========================================================================
#  real data only
# ===========================================================================

print("Real data only simulations")
# Load & prepare data
print(" Starting data preparation")
data_manager = DataManager(config=cfg.data)
data_manager.get_data()
print("Data is prepared")

results = []

for symbol in cfg.data.symbols:
    for model_name in cfg.pred.models:
        pred_model = ModelTrainer(model_name=model_name,
                                  n_steps_in=cfg.data.n_steps_in,
                                  n_steps_out=cfg.data.n_steps_out,
                                  pred_cfg=cfg_syn.pred)

        history = pred_model.train(
            x_train=data_manager.data[f'{symbol}_train_x'],
            y_train=data_manager.data[f'{symbol}_train_y'],
            x_val=data_manager.data[f'{symbol}_val_x'],
            y_val=data_manager.data[f'{symbol}_val_y'])

        test_results, y_pred = pred_model.evaluate(data_manager.data[f'{symbol}_test_x'],
                                           data_manager.data[f'{symbol}_test_y'])

        # Save predictions
        pred_dir = Path("data/predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        y_test = data_manager.data[f'{symbol}_test_y']
        
        # Inverse transform to get real values (log returns)
        scaler_y = data_manager.scalers[f"{symbol}_y"]
        y_test_inv = scaler_y.inverse_transform(y_test)
        y_pred_inv = scaler_y.inverse_transform(y_pred)
        
        df_preds = pd.DataFrame({
            'real': y_test_inv.flatten(),
            'pred': y_pred_inv.flatten()
        })
        # df_preds.to_csv(pred_dir / f"{symbol}_{model_name}_real_only.csv", index=False)

        results.append({
            "symbol": symbol,
            "data": "real_only",
            "data_weights": None,
            "model_name": model_name,
            "val_mse": min(history.history["val_loss"]),
            "test_mse": test_results[0],
            "test_mae": test_results[2],
            "test_mape": test_results[3]
        })
for syn_model in cfg_syn.data.syn_models:
    new_cfg = cfg_syn.data.model_copy(update={
            "syn_models": [syn_model]
    })
    data_manager_syn = DataManager(config=new_cfg)
    data_manager_syn.get_data()
    for symbol in cfg_syn.data.symbols:
        print(f"SYMBOL: {symbol}")
        x_mixed = data_manager_syn.data[f"{symbol}_train_x"]
        y_mixed = data_manager_syn.data[f"{symbol}_train_y"]

        real_train_x = data_manager.data[f"{symbol}_train_x"]
        n_real = len(real_train_x)

        for weight in [1, 10, 50]:
            print(f"WEIGHT: {weight}")

            sample_weights = np.ones(len(x_mixed))
            sample_weights[:n_real] = weight

            for model_name in cfg_syn.pred.models:
                print(f"MODEL: {model_name}")

                pred_model = ModelTrainer(model_name=model_name,
                                          n_steps_in=cfg_syn.data.n_steps_in,
                                          n_steps_out=cfg_syn.data.n_steps_out,
                                          pred_cfg=cfg_syn.pred)

                history = pred_model.train(
                    x_train=x_mixed,
                    y_train=y_mixed,
                    x_val=data_manager_syn.data[f'{symbol}_val_x'],
                    y_val=data_manager_syn.data[f'{symbol}_val_y'],
                    sample_weight=sample_weights)

                test_results, y_pred = pred_model.evaluate(data_manager_syn.data[f'{symbol}_test_x'],
                                                   data_manager_syn.data[f'{symbol}_test_y'])

                # Save predictions
                pred_dir = Path("data/predictions")
                pred_dir.mkdir(parents=True, exist_ok=True)
                
                y_test = data_manager_syn.data[f'{symbol}_test_y']
                
                scaler_y = data_manager_syn.scalers[f"{symbol}_y"]
                y_test_inv = scaler_y.inverse_transform(y_test)
                y_pred_inv = scaler_y.inverse_transform(y_pred)
                
                df_preds = pd.DataFrame({
                    'real': y_test_inv.flatten(),
                    'pred': y_pred_inv.flatten()
                })
                df_preds.to_csv(pred_dir / f"{symbol}_{model_name}_{syn_model}_w{weight}.csv", index=False)

                results.append({
                    "symbol": symbol,
                    "data": syn_model,
                    "data_weights": weight,
                    "model_name": model_name,
                    "val_mse": min(history.history["val_loss"]),
                    "test_mse": test_results[0],
                    "test_mae": test_results[2],
                    "test_mape": test_results[3]
                })

results_df = pd.DataFrame(results)
results_df.to_csv("analysis/results.csv")