# Import ML libraries
import tensorflow as tf
from tensorflow import feature_column as fc

import numpy as np

def arithmetic_scaling(data):
    data['pickup_longitude'] = (data['pickup_longitude'] + 75) / 8
    data['pickup_latitude'] =  (data['pickup_latitude'] - 40) / 8
    data['dropoff_longitude'] = (data['dropoff_longitude'] + 75) / 8
    data['dropoff_latitude'] = (data['dropoff_latitude'] - 40) / 8
    return data

# Haversine distance between pickup and dropoff location
def haversine_distance(data):

    # Earth's Radius in Kilometers
    radius = 6371.0

    # Convert latitude and longitude to radians
    p_lat = tf.math.radians(data['pickup_latitude'])
    p_lon = tf.math.radians(data['pickup_longitude'])
    d_lat = tf.math.radians(data['dropoff_latitude'])
    d_lon = tf.math.radians(data['dropoff_longitude'])

    # Calculate difference
    diff_lat = d_lat - p_lat
    diff_long = d_lon - p_lon

     # Apply Haversine formula
    a = (tf.math.sin(diff_lat / 2) ** 2 + 
         (tf.math.cos(p_lat) * 
          tf.math.cos(d_lat) * 
          tf.math.sin(diff_long / 2) ** 2)
    )
    c = 2 * tf.math.atan2(tf.math.sqrt(a), tf.math.sqrt(1 - a))

    data['distance'] = radius * c

    return data

def transform(inputs, col_names, num_buckets):
    
    input_data = inputs.copy(deep=True)
    input_data = input_data.map(
        lambda x: {k: x[k] for k in x if k != 'pickup_datetime'}
    )
    
    f_columns = {
        colname: tf.feature_column.numeric_column(colname)
        for colname in col_names
    }

    # Scale Latitude & Longitude Features
    input_data = input_data.map(arithmetic_scaling)

    # Calculate Haversine Distance Between Pickup & Dropoff Locations
    input_data = input_data.map(haversine_distance)

    f_columns['distance'] = fc.numeric_column('distance')

    # Create Bucketized Features
    latbuckets = np.linspace(0, 1, num_buckets).tolist()
    lonbuckets = np.linspace(0, 1, num_buckets).tolist()
    b_plat = fc.bucketized_column(
        f_columns['pickup_latitude'], latbuckets)
    b_dlat = fc.bucketized_column(
        f_columns['dropoff_latitude'], latbuckets)
    b_plon = fc.bucketized_column(
        f_columns['pickup_longitude'], lonbuckets)
    b_dlon = fc.bucketized_column(
        f_columns['dropoff_longitude'], lonbuckets)
    
    # Create Crossed Columns
    ploc = fc.crossed_column([b_plat, b_plon], num_buckets * num_buckets)
    dloc = fc.crossed_column([b_dlat, b_dlon], num_buckets * num_buckets)
    pd_pair = fc.crossed_column([ploc, dloc], num_buckets ** 4)

    f_columns['pickup_and_dropoff'] = fc.embedding_column(pd_pair, 100)

    return input_data, f_columns