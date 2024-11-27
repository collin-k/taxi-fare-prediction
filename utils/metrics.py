import tensorflow as tf

def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE)
    
    Args:
        y_true (tf.Tensor): True labels
        y_pred (tf.Tensor): Predicted values

    Returns:
        tf.Tensor: Computed RMSE
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE)

    Args:
        y_true (tf.Tensor): True labels
        y_pred (tf.Tensor): Predicted values

    Returns:
        tf.Tensor: Computed MAE
    """
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE)

    Args:
        y_true (tf.Tensor): True labels
        y_pred (tf.Tensor): Predicted values

    Returns:
        tf.Tensor: Computed MAPE as a percentage
    """
    error = tf.reduce_mean(
        tf.abs((y_true - y_pred) / 
        tf.maximum(tf.abs(y_true), tf.keras.backend.epsilon()))
    ) * 100
    return error

# R-squared (Coefficient of Determination)
def r_squared(y_true, y_pred):
    """
    Calculate R-squared

    Args:
        y_true (tf.Tensor): True labels
        y_pred (tf.Tensor): Predicted values

    Returns:
        tf.Tensor: Computed R-squared
    """
    total_variance = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    residual_variance = tf.reduce_sum(tf.square(y_true - y_pred))
    r2 = 1 - (
        residual_variance / 
        tf.maximum(total_variance, tf.keras.backend.epsilon())
    )
    return r2

def explained_variance(y_true, y_pred):
    """
    Calculate Explained Variance

    Args:
        y_true (tf.Tensor): True labels
        y_pred (tf.Tensor): Predicted values

    Returns:
        tf.Tensor: Computed Explained Variance, scaled between 0 and 1
    """
    numerator = tf.reduce_mean(tf.square(y_true - y_pred))
    denominator = tf.reduce_mean(tf.square(y_true - tf.reduce_mean(y_true)))
    exp_var =  1 - (
        numerator / 
        tf.maximum(denominator, tf.keras.backend.epsilon())
    )
    return exp_var
