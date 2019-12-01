import argparse
import gc
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

    # TRAIN_CONF['retrain'] = trial.suggest_int('retrain', 5, 30)
    # TRAIN_CONF['retrain_epochs'] = trial.suggest_int('retrain_epochs', 0, 40)

    # negative_rate = trial.suggest_uniform('sample_balance', 0.2, 0.95)
    # TRAIN_CONF['sample_balance'] = [negative_rate, 1.0 - negative_rate]
    # TRAIN_CONF['batch_size'] = trial.suggest_int('batch_size', 16, 128)
    # TRAIN_CONF['lr'] = trial.suggest_loguniform('lr', 1e-7, 1e-3)

    TRAIN_CONF['rnn_hidden_size'] = trial.suggest_int('rnn_hidden_size', 50, 400)
    TRAIN_CONF['rnn_n_layers'] = trial.suggest_int('rnn_n_layers', 1, 5)
    TRAIN_CONF['rnn_type'] = trial.suggest_categorical('rnn_type', ['gru', 'lstm', 'rnn'])


def objective(trial):

    dataset_cls = EEGDataSet
    set_dataloader_func = DATALOADERS[TRAIN_CONF['dataloader_type']]
    expt_note = 'Test Patient\tAccuracy\tRecall\n'

    metrics = [
        Metric('loss', direction='minimize', save_model=True),
        Metric('f1', direction='maximize'),
    ]

    TRAIN_CONF['class_names'] = list(set(LABELS.values()))
    TRAIN_CONF['reproduce'] = ''

    select_params(trial)

    train_manager = TrainManager(TRAIN_CONF, load_func, label_func, dataset_cls, set_dataloader_func, metrics, expt_note)
    model, val_metrics, test_metrics = train_manager.train_test()
    print(gc.collect())

    for phase, metrics in zip(['val', 'test'], [val_metrics, test_metrics]):
        for metric, value in metrics.items():
            trial.set_user_attr(f'{phase}_{metric}', value)
    return test_metrics['f1'].mean()


def optimize(study):
    """
    objectiveから返ってくる値を最小化するようn_trials回だけパラメータ更新を行う
    :return:
    """
    study.optimize(objective, n_trials=2, n_jobs=1)
    return study.trials_dataframe()


def tuning() -> float:

    now_time = datetime.today().strftime('%y%m%H%M')
    study = optuna.create_study(direction='minimize')
    for i in range(40):
        result_df = optimize(study)
        result_df.to_csv(f'trials_{now_time}.csv')
    print(study.best_params)


if __name__ == '__main__':
    assert TRAIN_CONF['train_path'] != '' or TRAIN_CONF['val_path'] != '', \
        'You need to select training, validation data file to training, validation in --train-path, --val-path argments'
    # returns loss or accuracy
    tuning()
