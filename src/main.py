from utils import setup_model_parameters
from dataloader import get_datasets
from model_runner import run

if __name__ == '__main__':
    # initialize variables, create directories, unzip dataset
    args = setup_model_parameters()
    # get datasets
    dataset_train, dataset_val, dataset_test = get_datasets()

    # run the model in either train, validation of test mode
    if args.train_condition:
        run(dataset_train, args, mode='train')
    if args.val_condition:
        run(dataset_val, args, mode='val')
    if args.test_condition:
        run(dataset_test, args, mode='test')