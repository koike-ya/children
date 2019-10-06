import pandas as pd
import argparse
from ml.src.metrics import Metric
from eeglibrary.src.eeg_dataloader import set_dataloader
from eeglibrary.src.eeg_dataset import EEGDataSet
from eeglibrary.src.preprocessor import preprocess_args
from eeglibrary import eeg
from ml.tasks.train_manager import TrainManager, train_manager_args
import torch


LABELS = {'none': 0, 'seiz': 1, 'arch': 2}


def train_args(parser):
    parser = train_manager_args(parser)
    parser = preprocess_args(parser)

    return parser


def label_func(path):
    return LABELS[path.split('/')[-1].replace('.pkl', '').split('_')[-1]]


def load_func(path):
    return torch.from_numpy(eeg.load_pkl(path).values.reshape(-1, ))


def experiment(train_conf) -> float:

    dataset_cls = EEGDataSet
    set_dataloader_func = set_dataloader

    metrics = [
        Metric('loss', direction='minimize'),
        Metric('accuracy', direction='maximize', save_model=True),
        Metric('recall_1', direction='maximize'),
        Metric('far', direction='minimize')
    ]

    train_conf['class_names'] = list(LABELS.values())
    train_manager = TrainManager(train_conf, load_func, label_func, dataset_cls, set_dataloader_func, metrics)

    train_manager.train()

    # if train_conf['train_manifest'] == 'all':
    #     for sub_name in subject_dir_names:
    #         args = arrange_paths(args, sub_name)
    #         train(args, class_names, label_func, metrics)
    # elif args.inference:
    #     pred_list, path_list = train(args, class_names, label_func, metrics)
    #     voting(args, pred_list, path_list)
    # else:
    #     train(args, class_names, label_func, metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    train_conf = vars(train_args(parser).parse_args())
    assert train_conf['train_path'] != '' or train_conf['val_path'] != '', \
        'You need to select training, validation data file to training, validation in --train-path, --val-path argments'
    # returns loss or accuracy
    experiment(train_conf)
