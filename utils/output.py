import matplotlib.pyplot as plt
from datetime import datetime

def arg_parsing(argument):
    """Parsing arguments passed in from command line"""

    exp_name_arg = argument.experiment_name
    if exp_name_arg is None:
        exp_name_arg = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    num_trials_arg = int(argument.num_trials)

    return exp_name_arg, num_trials_arg

def plot_performance(history, metrics):
    nrows = 1
    ncols = 2
    fig = plt.figure(figsize=(10, 5))

    for idx, key in enumerate(metrics):
        ax = fig.add_subplot(nrows, ncols, idx+1)
        plt.plot(history.history[key])
        plt.plot(history.history['val_{}'.format(key)])
        plt.title('model {}'.format(key))
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')