"""This module provides a class for configuring and using segmentation models.

It includes utilities for selecting different architectures, backbones,
input channels, number of classes,
number of filters, and weights to initialize the model.
"""

from tensorflow.keras import layers, models
from utils.transform import transform

class DNN_model():
    
    def __init__(self, model_configs):
        """Initialize the Deep Neural Network Model object with the provided model configuration.

        Parameters
        ----------
        model_config : dict
            A dictionary containing configuration parameters for the model.
            The dictionary should contain the following keys:
                - "model": str, The model type to use. Options are
                'unet', 'deeplabv3+', and 'fcn'.
                - "backbone": str, The encoder to use, which is the classification
                model that will be used to extract features. Options are
                listed on the smp docs.
                - "num_classes": int, The number of classes to predict. Should
                match the number of classes in the mask.
                - "weights": Union[str, bool], The weights to use for the model.
                If True, uses imagenet weights. Can also accept a string path
                to a weights file, or a WeightsEnum with pretrained weights.

        Returns:
        -------
        None
        """

        self.buckets = model_configs.get("buckets")
        self.feature_list = model_configs.get("features_list")
        self.loss_fn = model_configs.get("loss_fn")
        self.metrics = model_configs.get("metrics")
        self.nodes_list = model_configs.get("nodes_list")
        self.optimizer = model_configs.get("optimizer")
    
    def build(self):

        inputs = {
            colname: layers.Input(name=colname, shape=(), dtype='float32')
            for colname in self.feature_list
        }

        input_data, feature_cols = transform(
            inputs,
            self.feature_list,
            self.buckets)
        
        model_inputs = layers.DenseFeatures(feature_cols.values())(input_data)

        for l in range(len(self.nodes_list)):
            if l == 0:
                hidden = layers.Dense(
                    self.nodes_list[l], 
                    activation='relu', 
                    name='h1'
                )(model_inputs)
            else:
                hidden = layers.Dense(
                    self.nodes_list[l], 
                    activation='relu', 
                    name=f'h{l+1}'
                )(hidden)
        
        output = layers.Dense(1, activation='linear', name='fare')(hidden)

        model = models.Model(inputs, output)

        model.compile(
            optimizer=self.optimizer, 
            loss=self.loss_fn, 
            metrics=self.metrics)

        return model