import os
import sys

# Get the path of the script's directory
code_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.dirname(code_directory)
my_directory = os.path.dirname(os.path.dirname(project_directory))

path = os.path.join(code_directory)
sys.path.append(path)

from train import Trainer


def main():

    mode = "test"

    data_dict = {}

    #### directories ####
    data_dir = os.path.join(my_directory, 'data', 'BSD300', 'clean')
    
    data_dict['dir_train'] = os.path.join(data_dir, 'train')
    data_dict['dir_test'] =  os.path.join(data_dir, 'test')

    data_dict['dir_results'] = os.path.join(project_directory, 'results')
    data_dict['dir_checkpoints'] = os.path.join(project_directory, 'checkpoints')

    # adam optimizer
    data_dict['lr'] = 0.001
    data_dict['beta1'] = 0.5
    data_dict['beta2'] = 0.999

    # hyperparameters
    data_dict['num_epochs'] = 50
    data_dict['epoch_save_freq'] = 10
    data_dict['epoch_to_load'] = None

    TRAINER = Trainer(data_dict)

    if mode == "train":

        TRAINER.train()
    
    elif mode == "test":

        TRAINER.test()


if __name__ == '__main__':

    main()

