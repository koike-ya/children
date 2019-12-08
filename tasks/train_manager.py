import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from ml.models.adda_model_manager import AddaModelManager
from ml.models.keras_model_manager import KerasModelManager
from ml.models.model_manager import model_manager_args, BaseModelManager
from ml.src.dataloader import set_adda_dataloader
from ml.src.metrics import Metric

sys.path.append('..')
from src.const import LABELS, PHASES
from src.const import CHILDREN_PATIENTS
from src.const import CHBMIT_PATIENTS


ICTAL_WINDOW_SIZE = 15


def train_manager_args(parser):
    train_parser = parser.add_argument_group('train arguments')
    train_parser.add_argument('--only-test', action='store_true', help='Load learned model and not training')
    train_parser.add_argument('--k-fold', type=int, default=9,
                              help='The number of folds. 1 means training with whole train data')
    train_parser.add_argument('--cv-type', default='normal', help='Type of cross validation.',
                              choices=['normal', 'patient', 'ictal'])
    train_parser.add_argument('--test', action='store_true', help='Do testing, You should be specify k-fold with 1.')
    train_parser.add_argument('--infer', action='store_true', help='Do inference with test_path data,')
    train_parser.add_argument('--adda', action='store_true', help='Adversarial discriminative domain adaptation or not.')
    train_parser.add_argument('--data-type', default='children', choices=['children', 'chbmit'])
    train_parser.add_argument('--manifest-path', help='data file for training', default='input/train.csv')
    train_parser.add_argument('--only-one-patient', help='Only one patient ', action='store_true')
    train_parser.add_argument('--model-manager', default='pytorch')
    train_parser.add_argument('--retrain', type=int, default=0)
    parser = model_manager_args(parser)

    return parser


def label_func(path):
    return LABELS[path.split('/')[-1].replace('.pkl', '').split('_')[-1]]


def load_func(path):
    return torch.from_numpy(eeg.load_pkl(path).values.reshape(-1, ))


class TrainManager:
    def __init__(self, train_conf, load_func, label_func, dataset_cls, set_dataloader_func, metrics, expt_note=''):
        self.train_conf = train_conf
        self.load_func = load_func
        self.label_func = label_func
        self.dataset_cls = dataset_cls
        self.set_dataloader_func = set_dataloader_func
        self.metrics = metrics
        self.expt_note = expt_note
        self.data_dfs = pd.DataFrame()

    def memo_note(self, word, end=''):
        if self.expt_note:
            self.expt_note += word

    def _set_data_dfs(self):
        data_dfs = OrderedDict()

        if self.train_conf['data_type'] == 'children':
            if self.train_conf['only_one_patient']:
                data_dfs[0] = pd.read_csv(self.train_conf[f'manifest_path'], header=None)

            else:
                for patient in CHILDREN_PATIENTS:
                    path = Path(self.train_conf[f'manifest_path'])
                    manifest_name = f"{patient}_manifest.csv"
                    if not (path.parent / manifest_name).is_file():
                        continue
                    data_dfs[patient] = pd.read_csv(path.parent / manifest_name, header=None)

        elif self.train_conf['data_type'] == 'chbmit':
            patients = [Path(self.train_conf[f'manifest_path']).parent.parent.name] if self.train_conf['only_one_patient'] else CHBMIT_PATIENTS
            path = Path(self.train_conf['manifest_path'])
            for patient in patients:
                if (path.resolve().parents[2] / patient / str(path).split('/')[-2] / 'manifest.csv').is_file():
                    data_dfs[patient] = pd.read_csv(path.parents[2] / patient / 'interictal_preictal' / 'manifest.csv', header=None)

        else:
            raise NotImplementedError
            # for patient in PATIENTS:
            #     for phase in PHASES:
            #         path = self.train_conf[f'{phase}_path']
            #         data_df = pd.concat([data_df, pd.read_csv(path, header=None)])
        return data_dfs

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

    def _preictal_one_out_cv(self, fold_count, k):
        data_dfs = pd.concat(list(self.data_dfs.values()), axis=0)
        all_labels = data_dfs.squeeze().apply(lambda x: self.label_func(x))

        self.data_dfs = {}
        for class_ in [0, 1, 2]:
            self.data_dfs[class_] = data_dfs[all_labels == class_]

        # preictalを選ぶ
        ictal_start = data_dfs[all_labels == 2][0].apply(lambda x: int(x.split('/')[-1].split('_')[-3]))
        preictal_start_idxs = [0] + list(ictal_start[ictal_start - ictal_start.shift(1) != 256 * ICTAL_WINDOW_SIZE].index)
        print(preictal_start_idxs[:10])
        leave_out_preictal = self.data_dfs[1].loc[preictal_start_idxs[fold_count]:preictal_start_idxs[fold_count + 1], :]
        assert not leave_out_preictal.empty

        # interictalを選ぶ
        length = len(self.data_dfs[0]) // k
        start_index = fold_count * length
        leave_out_interictal = self.data_dfs[0].reset_index(drop=True).iloc[start_index:start_index + length, :]

        test_path_df = pd.concat([leave_out_preictal, leave_out_interictal]).reset_index(drop=True)

        train_val_pre = self.data_dfs[1][~self.data_dfs[1].index.isin(leave_out_preictal.index)].reset_index(drop=True)
        train_val_inte = self.data_dfs[0][~self.data_dfs[0].index.isin(leave_out_interictal.index)].reset_index(drop=True)

        train_path_df = pd.concat([train_val_pre.iloc[:len(train_val_pre) * 3 // 4, :],
                                   train_val_inte.iloc[:len(train_val_inte) * 3 // 4, :]]).reset_index(drop=True)
        val_path_df = pd.concat([train_val_pre.iloc[len(train_val_pre) * 3 // 4:, :],
                                   train_val_inte.iloc[len(train_val_inte) * 3 // 4:, :]]).reset_index(drop=True)

        return train_path_df, val_path_df, test_path_df

    def _ictal_one_out_cv(self, fold_count, k):

        data_dfs = pd.concat(list(self.data_dfs.values()), axis=0)
        all_labels = data_dfs.squeeze().apply(lambda x: self.label_func(x))

        self.data_dfs = {}
        for class_ in [0, 1]:
            self.data_dfs[class_] = data_dfs[all_labels == class_]

        # ictalを選ぶ
        ictal_start = data_dfs[all_labels == 1][0].apply(lambda x: int(x.split('/')[-1].split('_')[-3]))
        ictal_start_idxs = list(ictal_start[ictal_start - ictal_start.shift(1) != 500 * 15].index) + [1000000]
        leave_out_ictal = self.data_dfs[1].loc[ictal_start_idxs[fold_count]:ictal_start_idxs[fold_count + 1] - 1, :]
        print(leave_out_ictal.values[:10])
        assert not leave_out_ictal.empty

        # interictalを選ぶ
        length = len(self.data_dfs[0]) // k
        start_index = fold_count * length
        leave_out_interictal = self.data_dfs[0].reset_index(drop=True).iloc[start_index:start_index + length, :]

        test_path_df = pd.concat([leave_out_ictal, leave_out_interictal]).reset_index(drop=True)

        train_val_ictal = self.data_dfs[1][~self.data_dfs[1].index.isin(leave_out_ictal.index)].reset_index(drop=True)
        train_val_inte = self.data_dfs[0][~self.data_dfs[0].index.isin(leave_out_interictal.index)].reset_index(drop=True)

        train_path_df = pd.concat([train_val_ictal.iloc[:len(train_val_ictal) * 3 // 4, :],
                                   train_val_inte.iloc[:len(train_val_inte) * 3 // 4, :]]).reset_index(drop=True)
        val_path_df = pd.concat([train_val_ictal.iloc[len(train_val_ictal) * 3 // 4:, :],
                                 train_val_inte.iloc[len(train_val_inte) * 3 // 4:, :]]).reset_index(drop=True)

        return train_path_df, val_path_df, test_path_df

    def _normal_cv(self, fold_count, k):
        data_dfs = pd.concat(list(self.data_dfs.values()), axis=0)
        all_labels = data_dfs.squeeze().apply(lambda x: self.label_func(x))

        self.data_dfs = {}
        for class_ in self.train_conf['class_names']:
            self.data_dfs[class_] = data_dfs[all_labels == class_].reset_index(drop=True)

        test_path_df = pd.DataFrame()
        val_path_df = pd.DataFrame()
        train_path_df = pd.DataFrame()
        for class_, label_df in self.data_dfs.items():
            one_phase_length = len(label_df) // k
            start_index = fold_count * one_phase_length
            leave_out = label_df.iloc[start_index:start_index + one_phase_length, :]
            test_path_df = pd.concat([test_path_df, leave_out]).reset_index(drop=True)

            train_val_df = label_df[~label_df.index.isin(leave_out.index)].reset_index(drop=True)
            val_start_index = (fold_count % (k - 1)) * one_phase_length
            leave_out = train_val_df.iloc[val_start_index:val_start_index + one_phase_length, :]
            val_path_df = pd.concat([val_path_df, leave_out])

            train_path_df = pd.concat([train_path_df, train_val_df[~train_val_df.index.isin(leave_out.index)]])

        return train_path_df, val_path_df, test_path_df

    def _patient_one_out_cv(self, fold_count, k):
        patients = list(self.data_dfs.keys())
        n_patients_to_test = len(patients) // k
        assert float(n_patients_to_test) == len(patients) / k

        if self.train_conf['data_type'] == 'chbmit':
            data_dfs = self.data_dfs
            for (patient, df) in self.data_dfs.items():
                data_dfs[patient] = df[df[0].apply(lambda x: self.label_func(x)) != 2]
        
        test_patients = patients[fold_count * n_patients_to_test:(fold_count + 1) * n_patients_to_test]
        print(f'{test_patients} will be used as test patients')
        self.memo_note(test_patients)
        test_path_df = [self.data_dfs[patient] for patient in test_patients]
        test_path_df = pd.concat(test_path_df, axis=0, sort=False)

        train_val_patients = [patient for patient in patients if patient not in test_patients]
        val_start_idx = (fold_count % (k - 1)) * n_patients_to_test
        val_patients = train_val_patients[val_start_idx:val_start_idx + n_patients_to_test]
        print(f'{val_patients} will be used as validation patients')
        val_path_df = [self.data_dfs[patient] for patient in val_patients]
        val_path_df = pd.concat(val_path_df, axis=0, sort=False)
        
        train_path_df = [self.data_dfs[patient] for patient in train_val_patients if patient not in val_patients]
        train_path_df = pd.concat(train_path_df, axis=0, sort=False)

        return train_path_df, val_path_df, test_path_df

    def _train_test(self, model=None):
        # dataset, dataloaderの作成
        dataloaders = {}
        for phase in PHASES:
            dataset = self.dataset_cls(self.train_conf[f'{phase}_path'], self.train_conf,
                                       load_func=self.load_func, label_func=self.label_func, phase=phase)
            dataloaders[phase] = self.set_dataloader_func(dataset, phase, self.train_conf)

        model_manager = self._init_model_manager(dataloaders)

        model_manager.train(model)
        _, _, test_metrics = model_manager.test(return_metrics=True)

        return test_metrics, model_manager

    def _train_test_adda(self, whole_epoch=1):
        orig_model = None
        for epoch in range(whole_epoch):
            orig_metrics, orig_model = self._train_test(orig_model)

            # dataset, dataloaderの作成
            dataloaders = {}
            for domain, phase in zip(['source', 'target', 'test'], ['train', 'val', 'test']):
                dataset = self.dataset_cls(self.train_conf[f'{phase}_path'], self.train_conf, load_func=self.load_func,
                                           label_func=self.label_func)
                dataloaders[domain] = set_adda_dataloader(dataset, self.train_conf)

            model_manager = AddaModelManager(orig_model, self.train_conf, dataloaders, deepcopy(orig_metrics))

            model = model_manager.train()
            model_manager.model.fitted = True
            _, _, metrics = model_manager.test(return_metrics=True, load_best=False)

        return metrics, model_manager

    def _retrain(self, model_manager):
        # test_pathのfoldからnつをretrain用に、他をテストにまわして保存する
        n = self.train_conf['retrain'] * 60
        test_path_df = pd.read_csv(self.train_conf['test_path'])

        retrain_train_path = self.train_conf['test_path'].replace('_fold', '_retrain-train_fold')
        test_path_df.iloc[:n].to_csv(retrain_train_path, header=None, index=False)
        dataset = self.dataset_cls(retrain_train_path, self.train_conf, load_func=self.load_func,
                                   label_func=self.label_func)
        model_manager.dataloaders['retrain'] = self.set_dataloader_func(dataset, 'retrain', self.train_conf)

        retrain_test_path = self.train_conf['test_path'].replace('_fold', '_retrain-test_fold')
        test_path_df.iloc[n:].to_csv(retrain_test_path, header=None, index=False)
        dataset = self.dataset_cls(retrain_test_path, self.train_conf, load_func=self.load_func,
                                   label_func=self.label_func)
        model_manager.dataloaders['retrain_test'] = self.set_dataloader_func(dataset, 'retrain_test', self.train_conf)

        metrics = model_manager.retrain()

        for metric in metrics:
            if metric.name in ['accuracy', 'recall_1']:
                self.memo_note(f"\t{metric.average_meter['retrain_test'].best_score}")

        # 新しく作成したマニフェストファイルは削除
        [Path(path).unlink() for path in [retrain_train_path, retrain_test_path]]

    def _update_data_paths(self, fold_count: int, k: int):
        # fold_count...k-foldのうちでいくつ目か

        if self.train_conf['cv_type'] == 'normal':
            train_path_df, val_path_df, test_path_df = self._normal_cv(fold_count, k)
        elif self.train_conf['cv_type'] == 'patient':
            train_path_df, val_path_df, test_path_df = self._patient_one_out_cv(fold_count, k)
        elif self.train_conf['cv_type'] == 'ictal' and self.train_conf['data_type'] == 'children':
            train_path_df, val_path_df, test_path_df = self._ictal_one_out_cv(fold_count, k)
        elif self.train_conf['cv_type'] == 'ictal' and self.train_conf['data_type'] == 'chbmit':
            train_path_df, val_path_df, test_path_df = self._preictal_one_out_cv(fold_count, k)

        for phase in PHASES:
            file_name = f"{Path(self.train_conf['manifest_path'])}_{phase}_path_fold.csv"

            locals()[f'{phase}_path_df'].to_csv(file_name, index=False, header=None)
            self.train_conf[f'{phase}_path'] = str(file_name)
            print(f'{phase} data:\n', locals()[f'{phase}_path_df'][0].apply(self.label_func).value_counts())

    def _train_test_cv(self):
        orig_train_path = self.train_conf['train_path']

        if Path(self.train_conf['model_path']).is_file():
            Path(self.train_conf['model_path']).unlink()

        if self.train_conf['k_fold'] == 0:
            # データ全体で学習を行う
            raise NotImplementedError

        if self.train_conf['cv_type'] == 'ictal':
            if self.train_conf['data_type'] == 'children':
                ictal_label = 1
                ictal_interval = 500 * 15
            elif self.train_conf['data_type'] == 'chbmit':
                ictal_label = 1
                ictal_interval = 256 * ICTAL_WINDOW_SIZE

            data_dfs = pd.concat(list(self.data_dfs.values()), axis=0)
            all_labels = data_dfs.squeeze().apply(lambda x: self.label_func(x))
            ictal_start = data_dfs[all_labels == ictal_label][0].apply(lambda x: int(x.split('/')[-1].split('_')[-3]))
            self.train_conf['k_fold'] = ictal_start[ictal_start - ictal_start.shift(1) != ictal_interval].shape[0]
            print(f"{self.train_conf['k_fold']} folds")

        val_cv_metrics = {metric.name: np.zeros(self.train_conf['k_fold']) for metric in self.metrics}
        test_cv_metrics = {metric.name: np.zeros(self.train_conf['k_fold']) for metric in self.metrics}

        for i in range(self.train_conf['k_fold']):
            self._update_data_paths(i, self.train_conf['k_fold'])

            if self.train_conf['adda']:
                result_metrics, model_manager = self._train_test_adda()
            else:
                result_metrics, model_manager = self._train_test()

            print(f'Fold {i + 1} ended.')
            for metric in result_metrics:
                val_cv_metrics[metric.name][i] = metric.average_meter['val'].best_score
                test_cv_metrics[metric.name][i] = metric.average_meter['test'].best_score
                # print(f"Metric {metric.name} best score: {metric.average_meter['val'].best_score}")
                if metric.name in ['accuracy', 'recall_1']:
                    self.memo_note(f"\t{metric.average_meter['test'].best_score}")

            if self.train_conf['retrain']:
                self._retrain(model_manager)

            self.memo_note('\n')

        [print(f'{i + 1} fold {metric_name} score\t mean: {meter.mean() :.4f}\t std: {meter.std() :.4f}') for
         metric_name, meter in test_cv_metrics.items()]

        # 新しく作成したマニフェストファイルは削除
        [Path(self.train_conf[f'{phase}_path']).unlink() for phase in PHASES]

        return model_manager, val_cv_metrics, test_cv_metrics

    def test(self, model_manager=None) -> List[Metric]:
        if not model_manager:
            # dataset, dataloaderの作成
            dataloaders = {}
            dataset = self.dataset_cls(self.train_conf[f'test_path'], self.train_conf,
                                       load_func=self.load_func, label_func=self.label_func)
            dataloaders['test'] = self.set_dataloader_func(dataset, 'test', self.train_conf)

            model_manager = self._init_model_manager(dataloaders)

        return model_manager.test(return_metrics=True)

    def train_test(self):
        self.data_dfs = self._set_data_dfs()
        return self._train_test_cv()

    def infer(self) -> np.array:
        phase = 'infer'
        # dataset, dataloaderの作成
        dataloaders = {}
        dataset = self.dataset_cls(self.train_conf[f'test_path'], self.train_conf,
                                   load_func=self.load_func, label_func=self.label_func)
        dataloaders[phase] = self.set_dataloader_func(dataset, phase, self.train_conf)

        # modelManagerをインスタンス化、inferの実行
        model_manager = BaseModelManager(self.train_conf['class_names'], self.train_conf, dataloaders, self.metrics)
        return model_manager.infer()
