import os
import sys
sys.path.append(os.path.join(".."))

my_folder = os.path.join('/g', 'prevedel', 'members', 'Rauscher')
project_dir = os.path.join(my_folder, 'projects', 'N2N-BSD300-1')

from train import Trainer


def main():

    mode = "train"

    data_dict = {}

    #### directories ####
    data_dir = os.path.join(my_folder, 'data', 'BSD300', 'clean')
    
    data_dict['dir_train'] = os.path.join(data_dir, 'train')
    data_dict['dir_test'] =  os.path.join(data_dir, 'test')
    
    data_dict['dir_results'] = os.path.join(project_dir, 'results')
    data_dict['dir_checkpoints'] = os.path.join(project_dir, 'checkpoints')

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

