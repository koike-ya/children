import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
import argparse
from copy import deepcopy
from ml.src.metrics import Metric
from ml.models.model_manager import model_manager_args, BaseModelManager
from ml.models.keras_model_manager import KerasModelManager
from ml.src.metrics import AverageMeter
import torch

LABELS = {'none': 0, 'seiz': 1, 'arch': 2}
PHASES = ['train', 'val', 'test']
# 'MJ01128Z'は正例がないため除去
PATIENTS = ['YJ0112PQ', 'MJ00803P', 'YJ0100DP', 'YJ0100E9', 'MJ00802S', 'YJ01133T', 'YJ0112AU', 'WJ01003H', 'WJ010024']


def train_manager_args(parser):
    train_parser = parser.add_argument_group('train arguments')
    train_parser.add_argument('--only-test', action='store_true', help='Load learned model and not training')
    train_parser.add_argument('--k-fold', type=int, default=0,
                              help='The number of folds. 1 means training with whole train data')
    train_parser.add_argument('--test', action='store_true', help='Do testing, You should be specify k-fold with 1.')
    train_parser.add_argument('--infer', action='store_true', help='Do inference with test_path data,')
    train_parser.add_argument('--model-manager', default='pytorch')
    parser = model_manager_args(parser)

    return parser


def label_func(path):
    return LABELS[path.split('/')[-1].replace('.pkl', '').split('_')[-1]]


def load_func(path):
    return torch.from_numpy(eeg.load_pkl(path).values.reshape(-1, ))


class TrainManager:
    def __init__(self, train_conf, load_func, label_func, dataset_cls, set_dataloader_func, metrics):
        self.train_conf = train_conf
        self.load_func = load_func
        self.label_func = label_func
        self.dataset_cls = dataset_cls
        self.set_dataloader_func = set_dataloader_func
        self.metrics = metrics
        self.is_manifest = 'manifest' in self.train_conf['train_path']
        self.each_patient_df = self._set_each_patient_df()

    def _set_each_patient_df(self):
        each_patient_df = {}

        if self.is_manifest:
            for patient in PATIENTS:
                data_df = pd.DataFrame()
                for phase in PHASES:
                    path = Path(self.train_conf[f'{phase}_path'])
                    manifest_name = f"{patient}_{phase}_{path.name.split('_')[2]}"
                    data_df = pd.concat([data_df, pd.read_csv(path.parent / manifest_name, header=None)])
                each_patient_df[patient] = data_df
        else:
            raise NotImplementedError
            # for patient in PATIENTS:
            #     for phase in PHASES:
            #         path = self.train_conf[f'{phase}_path']
            #         data_df = pd.concat([data_df, pd.read_csv(path, header=None)])

        return each_patient_df

    def _init_model_manager(self, dataloaders):
        # modelManagerをインスタンス化、trainの実行
        if self.train_conf['model_manager'] == 'pytorch':
            model_manager = BaseModelManager(self.train_conf['class_names'], self.train_conf, dataloaders,
                                             deepcopy(self.metrics))
        elif self.train_conf['model_manager'] == 'keras':
            model_manager = KerasModelManager(self.train_conf['class_names'], self.train_conf, dataloaders,
                                              deepcopy(self.metrics))
        else:
            raise NotImplementedError

        return model_manager

    def _train_test(self):
        # dataset, dataloaderの作成
        dataloaders = {}
        for phase in PHASES:
            dataset = self.dataset_cls(self.train_conf[f'{phase}_path'], self.train_conf, phase,
                                       load_func=self.load_func, label_func=self.label_func)
            dataloaders[phase] = self.set_dataloader_func(dataset, phase, self.train_conf)

        model_manager = self._init_model_manager(dataloaders)

        model_manager.train()
        _, _, test_metrics = model_manager.test(return_metrics=True)

        return test_metrics, model_manager.model

    def _update_data_paths(self, fold_count: int, k: int):
        # fold_count...k-foldのうちでいくつ目か

        n_patients_to_test = len(PATIENTS) // k
        assert float(n_patients_to_test) == len(PATIENTS) / k
        test_patients = PATIENTS[fold_count * n_patients_to_test:(fold_count + 1) * n_patients_to_test]
        print(f'{test_patients} will be used as test patients')
        test_path_df = [df for patient, df in self.each_patient_df.items() if patient in test_patients]
        test_path_df = pd.concat(test_path_df, axis=0, sort=False)

        train_val_dfs = [df for patient, df in self.each_patient_df.items() if patient not in test_patients]
        train_val_dfs = pd.concat(train_val_dfs, axis=0, sort=False)

        train_path_df = pd.DataFrame()
        val_path_df = pd.DataFrame()

        # あとでラベルを結合するので、各患者毎にそれぞれのラベルを取り出して結合する
        for patient in PATIENTS:
            if patient in test_patients:
                continue
            df = self.each_patient_df[patient]
            all_labels = df.squeeze().apply(lambda x: self.label_func(x))
            for label in list(set(all_labels)):
                df_one_label = df[all_labels == label].reset_index(drop=True)
                train_path_df = pd.concat([train_path_df, df_one_label.iloc[:int(len(df_one_label) * 0.8)]])
                val_path_df = pd.concat([val_path_df, df_one_label.iloc[int(len(df_one_label) * 0.8):]])

        for phase in PHASES:
            file_name = self.train_conf[f'{phase}_path'][:-4].replace('_fold', '') + '_fold.csv'
            locals()[f'{phase}_path_df'].to_csv(file_name, index=False, header=None)
            self.train_conf[f'{phase}_path'] = file_name
            print(f'{phase} data:\n', locals()[f'{phase}_path_df'][0].apply(self.label_func).value_counts())

    def _train_test_k_fold(self):
        orig_train_path = self.train_conf['train_path']

        k_fold_metrics = {metric.name: np.zeros(self.train_conf['k_fold']) for metric in self.metrics}

        if self.train_conf['k_fold'] == 0:
            # データ全体で学習を行う
            raise NotImplementedError

        for i in range(self.train_conf['k_fold']):
            if Path(self.train_conf['model_path']).is_file():
                Path(self.train_conf['model_path']).unlink()

            self._update_data_paths(i, self.train_conf['k_fold'])

            result_metrics, model = self._train_test()

            print(f'Fold {i + 1} ended.')
            for metric in result_metrics:
                k_fold_metrics[metric.name][i] = metric.average_meter['test'].best_score
                # print(f"Metric {metric.name} best score: {metric.average_meter['val'].best_score}")

        [print(f'{i + 1} fold {metric_name} score\t mean: {meter.mean() :.4f}\t std: {meter.std() :.4f}') for
         metric_name, meter in k_fold_metrics.items()]

        # 新しく作成したマニフェストファイルは削除
        [Path(self.train_conf[f'{phase}_path']).unlink() for phase in PHASES]

        return model, k_fold_metrics

    def test(self, model_manager=None) -> List[Metric]:
        if not model_manager:
            # dataset, dataloaderの作成
            dataloaders = {}
            dataset = self.dataset_cls(self.train_conf[f'test_path'], self.train_conf, phase='test',
                                       load_func=self.load_func, label_func=self.label_func)
            dataloaders['test'] = self.set_dataloader_func(dataset, 'test', self.train_conf)

            model_manager = self._init_model_manager(dataloaders)

        return model_manager.test()

    def train_test(self):
        if not self.train_conf['only_test']:
            return self._train_test_k_fold()
        else:
            self.test()

    def infer(self) -> np.array:
        phase = 'infer'
        # dataset, dataloaderの作成
        dataloaders = {}
        dataset = self.dataset_cls(self.train_conf[f'test_path'], self.train_conf, phase=phase,
                                   load_func=self.load_func, label_func=self.label_func)
        dataloaders[phase] = self.set_dataloader_func(dataset, phase, self.train_conf)

        # modelManagerをインスタンス化、inferの実行
        model_manager = BaseModelManager(self.train_conf['class_names'], self.train_conf, dataloaders, self.metrics)
        return model_manager.infer()
