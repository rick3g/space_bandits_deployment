import argparse
import json
import logging
import os
# import sagemaker_containers
import sys
from pathlib import Path
import pandas as pd
import torch
from space_bandits import LinearBandits, NeuralBandits, load_model

TRAIN_CHANNEL = "training"
EVAL_CHANNEL = "evaluation"
MODEL_CHANNEL = "pretrained_model"
MODEL_OUTPUT_DIR = os.environ.get('SM_MODEL_DIR', "/opt/ml/model")
MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, "vw.model")
DATA_OUTPUT_DIR = "/opt/ml/output/data"
DATA_EXTENSIONS = ['csv']

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def check_extensions(fname, extensions):
    for extension in extensions:
        if fname.endswith('.' + extensions):
            return True

    return False

def get_files_from_dir(dir_path, extensions=DATA_EXTENSIONS):
    dir_path = Path(dir_path)
    file_paths = [dir_path/f for f in os.listdir(dir_path) if check_extensions(f,extensions)]
    return file_paths

def get_data(data_path, choose_file):
    '''
    Function to get a dataframe for training the model from a path
    Args:
        data_path: path given for data
        choose_file: function to be applied to choose the file if there are multiple files in the directory
    Returns:
        df: dataframe created from the chosen file
    '''
    data_files = get_files_from_dir(data_path)
    if len(data_files) == 0:
        raise Exception('There are no data files in this directory')
    if len(data_files) > 1:
        file = choose_file(data_files)
    with open(file, 'r') as f:
        df = pd.read_csv(f)
    return df

def get_dim_contexts_actions(df):
    pass

def choose_first(files):
    return files[0]

def split_data(df, context_dim, actions_dim):
    contexts = df[:,:context_dim]
    actions = df[:,context_dim:context_dim+actions_dim]
    rewards = df[:,context_dim+actions_dim:]
    return contexts,actions,rewards

def train(args):
    '''
    Method to train the model. Gets called when you create an estimator and call fit on it passing in the inputs.
    Args:
        args: object containing the arguments

    '''

    use_cuda = args.num_gpus > 0
    device = torch.device('cuda' if use_cuda else 'cpu')

    channel_names = json.loads(os.environ['SM_CHANNELS'])
    hyperparameters = json.loads(os.environ['SM_HPS'])

    # Fetch algorithm hyperparameters
    action_dim = int(hyperparameters.get('action_dim', 0))
    context_dim = int(hyperparameters.get('context_dim', 0 ))
    model_type = str(hyperparameters.get("model_type", 'neural'))
    model_file = str(hyperparameters.get("mode_file", 'model.pickle'))

    # Load the data to use for training
    training_data = get_data(channel_names['SM_CHANNEL_TRAINING'], choose_first)

    # If action_dim or context_dim = 0 then we will get the dims from the dataframe
    if action_dim == 0 or context_dim == 0:
        context_dim, action_dim = get_dim_contexts_actions(training_data)




    if MODEL_CHANNEL in channel_names:

        # Load the pre-trained model for training.
        model_folder = os.environ[f'SM_CHANNEL_{MODEL_CHANNEL.upper()}']
        model_path = Path(model_folder)/model_file
        logging.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = load_model(f)

    else:
        logging.info(f"No pre-trained model has been specified in channel {MODEL_CHANNEL}."
                     f"Training will start from scratch.")
        # Create model based on model_type hyperparameter
        if model_type == 'linear':
            model = LinearBandits(actions_dim,context_dim)
        elif model_type == 'neural':
            model = NeuralBandits(actions_dim,context_dim)
        else:
            raise Exception('%s is not a supported model type' % model_type)


    # Get the contexts, actions, and rewards to fit the model
    contexts, actions, rewards = split_data(training_data, context_dim, action_dim)
    model.fit(contexts,actions,rewards)

    # Save model
    model.save(model_dir/model_file)


def get_hparams(parser):
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    return parser



def save_model(model, model_dir, fname='model.pth'):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, fname)
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser = get_hparams(parser)

    # Container environment
    # parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    # parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    # parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    # parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())
