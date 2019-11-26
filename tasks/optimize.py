import argparse
from datetime import datetime

import optuna
import torch
from eeglibrary import eeg
from eeglibrary.src.eeg_dataloader import set_dataloader as eeg_dataloader
from eeglibrary.src.eeg_dataset import EEGDataSet
from eeglibrary.src.preprocessor import preprocess_args
from ml.src.dataloader import set_dataloader, set_ml_dataloader
from ml.src.metrics import Metric
from train_manager import TrainManager, train_manager_args

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


parser = argparse.ArgumentParser(description='train arguments')
TRAIN_CONF = vars(train_args(parser).parse_args())


def label_func(path):
    return LABELS[path.split('/')[-1].replace('.pkl', '').split('_')[-1]]
    # return PATIENTS.index(path.split('/')[-2])


def load_func(path):
    return torch.from_numpy(eeg.load_pkl(path).values.reshape(-1, ))


def select_params(trial):

    # TRAIN_CONF['iterations'] = trial.suggest_int('iterations', 10, 100)
    # TRAIN_CONF['k_disc'] = trial.suggest_int('k_disc', 5, 30)
    # TRAIN_CONF['k_clf'] = trial.suggest_int('k_clf', 5, 30)

    TRAIN_CONF['retrain'] = trial.suggest_int('retrain', 5, 30)
    TRAIN_CONF['retrain_epochs'] = trial.suggest_int('retrain_epochs', 0, 40)

    # negative_rate = trial.suggest_uniform('sample_balance', 0, 1.0)
    # TRAIN_CONF['sample_balance'] = [negative_rate, 1.0 - negative_rate]
    # TRAIN_CONF['batch_size'] = trial.suggest_int('batch_size', 16, 128)
    # TRAIN_CONF['lr'] = trial.suggest_loguniform('lr', 1e-7, 1e-3)


def objective(trial):

    dataset_cls = EEGDataSet
    set_dataloader_func = DATALOADERS[TRAIN_CONF['dataloader_type']]
    expt_note = 'Test Patient\tAccuracy\tRecall\n'

    metrics = [
        Metric('loss', direction='minimize', save_model=True),
        Metric('accuracy', direction='maximize'),
        Metric('recall_1', direction='maximize'),
        Metric('far', direction='minimize')
    ]

    TRAIN_CONF['class_names'] = list(set(LABELS.values()))

    select_params(trial)

    train_manager = TrainManager(TRAIN_CONF, load_func, label_func, dataset_cls, set_dataloader_func, metrics, expt_note)
    model, val_metrics, test_metrics = train_manager.train_test()
    for phase, metrics in zip(['val', 'test'], [val_metrics, test_metrics]):
        for metric, value in metrics.items():
            trial.set_user_attr(f'{phase}_{metric}', value)
    return val_metrics['loss'].mean()


def tuning() -> float:

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    now_time = datetime.today().strftime('%y%m%H%M')
    study.trials_dataframe().to_csv(f'trials_{now_time}.csv')


if __name__ == '__main__':
    assert TRAIN_CONF['train_path'] != '' or TRAIN_CONF['val_path'] != '', \
        'You need to select training, validation data file to training, validation in --train-path, --val-path argments'
    # returns loss or accuracy
    tuning()
