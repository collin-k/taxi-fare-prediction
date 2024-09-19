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

**Pickup Location**: POINT (-73.99902 40.73915)

**Dropoff Location**: POINT (-73.990369, 40.717866)

### Cleaning

#### Location-based columns


# Exploratory Data Analysis

## Initial Data Exploration

### Fare & Datetime Distributions, Fare Amounts by Datetime, Mean Fares by the Day of the Week
<img src="https://github.com/collin-k/taxi-fare-prediction/blob/main/visualizations/quad-graphs.png">

#### Key Takeaways
* The fare_amounts column is right-skewed with a mean of $11.35 and standard deviation of $9.98.
* There appears to be peaks in the pickup datetime distribution, but there does not seem be a conclusive pattern.
* There does not seem be a relationship between the price and pickup datetime.
* There does not seem be a relationship between the price and the week of day of the trip.

### Latitude & Longitude Distributions
<img src="https://github.com/collin-k/taxi-fare-prediction/blob/main/visualizations/location_graphs.png">

#### Key Takeaways
* The latitude columns are moderately normally distributed while the longitude columns are right-skewed.

### Pickup & Dropoff Locations Before Cleaning
<img src="https://github.com/collin-k/taxi-fare-prediction/blob/main/visualizations/pickup_locs.png">
<img src="https://github.com/collin-k/taxi-fare-prediction/blob/main/visualizations/dropoff_locs.png">

#### Key Takeaways
After visually analyzing the geospatial data, the dataset was cleaned using the following conditions:
* Pickup and Dropoff locations must be in sensible range of New York City Boroughs (outliers were removed)
* Pickup and Dropoff locations must be different from one another (trips with idential pickup and dropoff locations were removed)
* Pickup and Dropoff locations must be on land (locations in bodies of water were removed)

### Pickup & Dropoff Locations After Cleaning
<img src="https://github.com/collin-k/taxi-fare-prediction/blob/main/visualizations/clean_pickup_locs.png">
<img src="https://github.com/collin-k/taxi-fare-prediction/blob/main/visualizations/clean_dropoff_locs.png">

#### Key Takeaways
* The vast majority of pickup and dropoff locations are in Manhattan, Brooklyn, and Queens.

### Pickup & Dropoff Heatmap for NYC Boroughs
<img src="https://github.com/collin-k/taxi-fare-prediction/blob/main/visualizations/nyc_pickups.png">
<img src="https://github.com/collin-k/taxi-fare-prediction/blob/main/visualizations/nyc_dropoffs.png">

#### Key Takeaways
* The majority of pickups and dropoffs within NYC boroughs are in Manhattan and East New York (in Brooklyn).

# Model Development

### Baseline Model
#### Description & Initial Results
To investigate the dataset, I built a baseline model using minimal feature engineering. The pickup_datetime column was parsed to extract and create dummy columns for year, month, hour of pickup. Then, the distance traveled was calculating using 'Manhattan Distance.' Lastly, the latitude and longitude columns were scaled manually for better model performance. The model consisted of two (2) hidden layers.

**Test loss (MSE):** 106.10

**Test RMSE:** 8.92

#### Key Takeaways
As discovered during data analysis, the mean fare of the dataset is $11.35. This indicated that the baseline model, with a root mean squared error (RMSE) of $8.92, performs poorly. The next steps were to adjust my feature engineering methods and model architecture.

### Final Model
#### Description & Results
As discovered during data analysis, the there was no clear relationship between the pickup_datetime and fare_amount columns. The pickup_datetime column, therefore, was removed from the list of features. To test for model performance, distance traveled was calculated using two (2) alternative methods (Harversine, Euclidean). Then, two (2) alternative methods of coordinate scaling were used (scikit-learn's MinMaxScaler() and StandardScaler()). To account for minor differences between different pickup and dropoff locations, the latitude and longitude values were bucketized to, then, create crossed columns. After testing the model with a different number of hidden layers and nodes per layers, the final model consisted of two (2) hidden layers with sixty-four (64) and thirty-two (32) nodes, respectively. 

**Test Loss (MSE):** 35.06

**Test RMSE:** 3.82
