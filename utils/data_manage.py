import pandas as pd
import tensorflow as tf
from configs import config
from sklearn.model_selection import train_test_split

def clean_data(data):
    
    data.drop(
        columns=['pickup_datetime', 'passenger_count', 'id'], 
        inplace=True,
    )

    # Remove Extreme Outliers & Trips With Identical Pickup and Dropoff Locations
    data.drop(
        data[
            (data['pickup_lat'] == data['dropoff_lat'])
            & (data['pickup_lon'] == data['dropoff_lon'])
        ].index,
        inplace=True,
    )
    data.drop(
        data[
            (data['pickup_lat'] >= 41) | (data['dropoff_lat'] >= 41)
        ].index,
        inplace=True,
    )
    data.drop(
        data[
            (data['pickup_lon'] >= -73.6) | (data['dropoff_lon'] >= -73.6)
        ].index,
        inplace=True,
    )
    data.drop(
        data[
            (data['pickup_lat'] <= 40.5) | (data['dropoff_lat'] <= 40.5)
        ].index,
        inplace=True,
    )

    # Remove Trips with Pickup or Dropoff Locations In Bodies of Water
    data.drop(index=[2991,5309,10251], inplace=True)
    
    return data

def split_data(data):
    
    y = data['fare_amount'].copy()
    X = data.drop(columns=['fare_amount'])

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def to_dataset(X, y, shuffle=False, batch_size=16, num_epochs=10):

    X_dict = X.to_dict(orient='list')

    dataset = tf.data.Dataset.from_tensor_slices((X_dict,y))
    if shuffle:
        dataset = dataset.shuffle(1000)
        dataset = dataset.prefetch(1).repeat()
    dataset = dataset.batch(batch_size).repeat(num_epochs)

    return dataset

def load_data():
    df_cols = [
        'fare_amount',
        'pickup_datetime',
        'pickup_lon', 
        'pickup_lat', 
        'dropoff_lon', 
        'dropoff_lat',
        'passenger_count',
        'id',
    ]
    data = pd.read_csv(
        config.TAXI_DATA, 
        names=df_cols, 
        header=None
    )
    
    return data

def prepare_data():
    
    data = load_data()
    data = clean_data(data)
    train, valid, test = split_data(data)
    
    return train, valid, test