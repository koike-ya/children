import pandas as pd
import argparse
from ml.src.metrics import Metric
from eeglibrary.src.eeg_dataloader import set_dataloader
from eeglibrary.src.eeg_dataset import EEGDataSet
from eeglibrary.src.preprocessor import preprocess_args
from eeglibrary.src.eeg import EEG
from train_manager import TrainManager, train_manager_args
import torch


LABELS = {'interictal': 0, 'preictal': 1, 'ictal': 2}


def train_args(parser):
    parser = train_manager_args(parser)
    parser = preprocess_args(parser)
    expt_parser = parser.add_argument_group("Experiment arguments")
    expt_parser.add_argument('--reproduce', help='Method name for reproduction', default='')

    return parser


def label_func(path):
    return LABELS[path.split('/')[-1].replace('.pkl', '').split('_')[-1]]


def load_func(path):
    return torch.from_numpy(EEG.load_pkl(path).values.reshape(-1, ))


def experiment(train_conf) -> float:

    dataset_cls = EEGDataSet
    set_dataloader_func = set_dataloader
    expt_note = 'Test Patient\tAccuracy\tRecall\n'

    metrics = [
        Metric('loss', direction='minimize'),
        Metric('accuracy', direction='maximize', save_model=True),
        Metric('recall_1', direction='maximize'),
        Metric('far', direction='minimize')
    ]

    train_conf['class_names'] = [0, 1, 2]
    train_conf['model_manager'] = 'keras'
    train_manager = TrainManager(train_conf, load_func, label_func, dataset_cls, set_dataloader_func, metrics, expt_note)

    train_manager.train_test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    train_conf = vars(train_args(parser).parse_args())
    assert train_conf['train_path'] != '' or train_conf['val_path'] != '', \
        'You need to select training, validation data file to training, validation in --train-path, --val-path argments'
    # returns loss or accuracy
    experiment(train_conf)
