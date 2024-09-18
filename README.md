# Predicting Taxi Fares with DNN

The first part of this notebook displays the exploratory data analysis of taxi fare data. The second part of this notebook displays my baseline model and final DNN model used to predict New York City taxi fares.

# Metadata

Raw data can be accessed on [this github page](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/toy_data). The dataset consisted of 7 columns and 10,476 rows. The following table provides descriptions for each column, including the column name, data type, and an example.

| **Column Name** | **Data Type**  | **Description**  | **Example**  |
|---|---|---|---|
| _fare_amount_  | `float64`  | Total fare for trip |  11.3 |
| _pickup_datetime_ | `object` | Date & Time of Pickup  | 2011-01-28 20:42:59 UTC  |
| _pickup_longitude_ | `float64` | Longitude of Pickup  | -73.999022 |
| _pickup_latitude_  | `float64` | Latitude of Pickup | 40.739146 |
| _dropoff_longitude_  | `float64` | Longitude of Dropoff |  -73.990369  |
| _dropoff_latitude_  | `float64` | Latitude of Dropoff  | 40.717866 |
| _passenger_count_ | `int64`  | Number of Passengers  | 1 |

# Data Wrangling

### Data Type Conversion

#### Pickup_datetime
As the dataset is meant to represent taxi fares in New York City, the values of the timezone for values in the pickup_datetime column needed to be converted from UTC to EST/EDT using the _pytz_ library. However, as the column was originally of data type object, so the column was first converted to the datetime type.

#### Location-based columns
To better analyze the pickup and dropoff points in the dataset, I converted each location into Point data types using the shapely.geometry library. The following is an example of the pickup and dropoff locations:

### Cleaning

#### Location-based columns

**Pickup Location**: POINT (-73.99902 40.73915)

**Dropoff Location**: POINT (-73.990369, 40.717866)

# Exploratory Data Analysis



The fare_amounts column is right-skewed with a mean of $11.35 and standard deviation of $9.98.



# Model Development

 

