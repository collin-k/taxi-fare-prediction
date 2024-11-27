"""Trains a DNN model.

To run: from repo directory (taxi-fare-prediction)
> python train.py configs.<config> [--experiment_name <name>]
    [--num_trials <num>]
"""

import argparse
import importlib.util
import logging
from configs import config
from datetime import datetime
from model import DNN_model
from utils.metrics import rmse
from utils.output import arg_parsing
from utils.data_manage import prepare_data

def build_model():
    """Setting up training model, loss function and measuring metrics

    Returns:
        tuple: A tuple containing:
            - model: The PyTorch model instance.
            - loss_fn: The loss function to use for training.
            - train_jaccard: The metric to measure Jaccard index on the training set.
            - test_jaccard: The metric to measure Jaccard index on the test set.
            - jaccard_per_class: The metric to measure Jaccard index per class.
            - optimizer: The optimizer for training the model.
    """
    
    # Assign Model Configs
    model_configs = {
        "buckets": config.NUM_BUCKETS,
        "feature_list": config.FEATURES,
        "loss_fn": config.LOSS_FN,
        "metrics": [rmse, 'mse'],
        "nodes_list": config.NODES,
        "optimizer": config.OPTIMIZER,
    }

    # Create Model
    logging.info("Building the model...")
    model = DNN_model(model_configs).build()
    
    #logging.info(model)

    return model

def test(model, test_data):
    
    logging.info("Training complete. Evaluating on test data...")
    test_loss, test_rmse, test_mse = model.evaluate(test_data)
    logging.info(f"Test Loss: {test_loss}, Test RMSE: {test_rmse}")
    return test_loss, test_rmse, test_mse

def train(model, train_data, valid_data):
    
    logging.info(f"Starting training: {args.experiment_name}")
    history = model.fit(
        train_data, 
        validation_data=valid_data,
        epochs=config.EPOCHS,
        steps_per_epoch=(len(train) // 16),
        verbose=0
    )
    return

def one_trial(exp_name, num):
    
    logging.info("Preparing data...")
    train_data, valid_data, test_data = prepare_data()
    model = build_model()

    train(model, train_data, valid_data)

    test_loss, test_rmse, test_mse = test(model, test_data)

    return test_loss, test_rmse, test_mse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DNN model to predict taxi fare"
    )
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of experiment",
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument(
        "--num_trials",
        type=str,
        help="Please enter the number of trials for each train",
        default="1",
    )

    args = parser.parse_args()
    config_module = importlib.import_module(args.config)
    exp_name, num_trials = arg_parsing(args)

    
    def run_trials():
        for num in range(num_trials):
            test_loss, test_rmse, test_mse = one_trial(exp_name, num,)
            print(test_loss, test_rmse, test_mse)
    
    run_trials()