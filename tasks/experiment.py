import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from ml.src.metrics import Metric
from eeglibrary.src.eeg_dataloader import set_dataloader as eeg_dataloader
from ml.src.dataloader import set_dataloader, set_ml_dataloader
from eeglibrary.src.eeg_dataset import EEGDataSet
from eeglibrary.src.preprocessor import preprocess_args
from eeglibrary import eeg
from train_manager import TrainManager, train_manager_args
import torch
import json


LABELS = {'none': 0, 'seiz': 1}
PATIENTS = ['YJ0112PQ', 'MJ00803P', 'YJ0100DP', 'YJ0100E9', 'MJ00802S', 'YJ01133T', 'YJ0112AU', 'WJ01003H', 'WJ010024']
DATALOADERS = {'normal': set_dataloader, 'eeg': eeg_dataloader, 'ml': set_ml_dataloader}


def train_args(parser):
    parser = train_manager_args(parser)
    parser = preprocess_args(parser)
    expt_parser = parser.add_argument_group("Experiment arguments")
    expt_parser.add_argument('--expt-id', help='data file for training', default='')
    expt_parser.add_argument('--dataloader-type', help='Dataloader type.', choices=['normal', 'eeg', 'ml'], default='eeg')

    return parser


def label_func(path):
    return LABELS[path.split('/')[-1].replace('.pkl', '').split('_')[-1]]
    # return PATIENTS.index(path.split('/')[-2])


def load_func(path):
    return torch.from_numpy(eeg.load_pkl(path).values.reshape(-1, ))


def experiment(train_conf) -> float:

    dataset_cls = EEGDataSet
    set_dataloader_func = DATALOADERS[train_conf['dataloader_type']]
    expt_note = 'Test Patient\tAccuracy\tRecall\n'

    metrics = [
        Metric('loss', direction='minimize', save_model=True),
        Metric('accuracy', direction='maximize'),
        Metric('recall_1', direction='maximize'),
        # Metric('far', direction='minimize')
    ]

    train_conf['class_names'] = list(set(LABELS.values()))
    train_manager = TrainManager(train_conf, load_func, label_func, dataset_cls, set_dataloader_func, metrics, expt_note)

    model, metrics = train_manager.train_test()

    now_time = datetime.today().strftime('%y%m%H%M')
    expt_name = f"{len(train_conf['class_names'])}-class_{train_conf['model_type']}_{train_conf['expt_id']}_{now_time}.txt"
    with open(Path(__file__).parent.parent / 'output' / expt_name, 'w') as f:
        f.write(f"experiment notes:\n{train_manager.expt_note}\n\n")
        f.write(f"{train_conf['k_fold']} fold results:\n")
        for metric_name, meter in metrics.items():
            f.write(f'{metric_name} score\t mean: {meter.mean() :.4f}\t std: {meter.std() :.4f}\n')
        f.write('\nParameters:\n')
        f.write(json.dumps(train_conf, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    train_conf = vars(train_args(parser).parse_args())
    assert train_conf['train_path'] != '' or train_conf['val_path'] != '', \
        'You need to select training, validation data file to training, validation in --train-path, --val-path argments'
    # returns loss or accuracy
    experiment(train_conf)
