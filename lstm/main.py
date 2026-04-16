# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import os
import json

import config
from data_handler import DataHandler
from models import CNN_1LSTM, CNN_2LSTM, CNN
from trainer import Trainer
import argparse

def write_config_file(training_id, config_module):
    """
    Intelligently saves only the uppercase configuration variables from the config module.
    """
    os.makedirs("Configs/SPARL3", exist_ok=True)
    
    # Create a dictionary of only the uppercase settings from the config file
    config_dict = {key: getattr(config_module, key) for key in dir(config_module) if key.isupper()}

    with open(f"Configs/SPARL3/config_{training_id}.json", "w") as f:
        # The dictionary is now clean and can be saved directly
        json.dump(config_dict, f, indent=4)
        
    print(f"Config for run {training_id} saved.")

def single_run(use_acc= True, use_gyro=True, use_baro=True, test_experiment_id=None, validation_experiment_id=None):

    if not use_acc:
        # remove accelerometer columns from sensor list
        config.SENSOR_COLS.remove("Acc.x")
        config.SENSOR_COLS.remove("Acc.y")
        config.SENSOR_COLS.remove("Acc.z")
        config.IN_CHANNELS -= 3
    if not use_gyro:
        # remove gyroscope columns from sensor list
        config.SENSOR_COLS.remove("Gyro.x")
        config.SENSOR_COLS.remove("Gyro.y")
        config.SENSOR_COLS.remove("Gyro.z")
        config.IN_CHANNELS -= 3
    if not use_baro:
         # remove barometer column from sensor list
         config.SENSOR_COLS.remove("Baro.x")
         config.IN_CHANNELS -= 1
    if test_experiment_id is not None:
        config.TEST_EXPERIMENT_ID = test_experiment_id
    if validation_experiment_id is not None:
        config.VALIDATION_EXPERIMENT_ID = validation_experiment_id
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mode_suffix = "_superclass_training" if config.USE_SUPERCLASSES else "_allclass_training"
    training_id = f"{timestamp}{mode_suffix}_single_run"
    csv_path = f"sparl3_csv/summary_{training_id}.csv"

    data_handler = DataHandler(config)
    train_loader, val_loader, test_loader, le, class_weights = data_handler.get_data_loaders()

    num_classes = len(le.classes_)
    
    if config.MODEL_NAME == "CNN_1LSTM":
        model = CNN_1LSTM(config.NUM_CNN_LAYERS, num_classes, config.IN_CHANNELS, lstm_hidden_size=config.LSTM_HIDDEN_SIZE)
    elif config.MODEL_NAME == "CNN_2LSTM":
        model = CNN_2LSTM(config.NUM_CNN_LAYERS, num_classes, config.IN_CHANNELS)
    elif config.MODEL_NAME == "CNN":
        model = CNN( config.IN_CHANNELS, num_classes)
    else:
        raise ValueError(f"Unknown model name: {config.MODEL_NAME}")
    model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        test_loader=test_loader, le=le, optimizer=optimizer,
        criterion=criterion, config=config
    )

    trainer.train(training_id)
    summary = trainer.evaluate(training_id, csv_path)
    
    write_config_file(training_id, config)
    print("Final MCC on test set:", summary['mcc'])
    print("Final accuracy on test set:", summary['test_acc'])
    print("Final F1-macro on test set:", summary['test_f1_macro'])
    return summary

def reset_config():
    config.SENSOR_COLS = ["Acc.x", "Acc.y", "Acc.z", "Gyro.x", "Gyro.y", "Gyro.z", "Baro.x"]
    config.IN_CHANNELS = len(config.SENSOR_COLS)

if __name__ == "__main__":
    # get argument for run id from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="id of gpu to use")
    args = parser.parse_args()

    if args.gpu_id:
        GPU_ID = args.gpu_id
        config.DEVICE = f"cuda:{GPU_ID}"
        print(f"This training process will run on GPU: {GPU_ID}")
    for folder in ["Models/SPARL3", "sparl3_csv", "Plots/SPARL3", "Pred/SPARL3", "Configs/SPARL3"]:
        os.makedirs(folder, exist_ok=True)
    single_run(use_acc=True, use_gyro=True, use_baro=False)