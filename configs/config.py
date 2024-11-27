"""Configuration parameters for model training and data processing

Contains paths to data directories, model architecture settings, 
hyperparameters, and options used throughout training pipeline.
"""

import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING logs

DATA_ROOT = "~/taxi-fare-prediciton/data"
OUTPUT_ROOT = "~/taxi-fare-prediction/plots"
TAXI_DATA = str(Path(DATA_ROOT) / "taxi_fare.csv")
NYC_SHAPE_DATA = str(Path(DATA_ROOT) / "nyc_shape.shp")

# Model Parameters
BATCH_SIZE = 16
EPOCHS = 20
LOSS_FN = 'mse'
NUM_BUCKETS = 10
OPTIMIZER = 'adam'

NODES = [
    64,
    32,
    #16,
    #8,
]

FEATURES = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "distance",
]