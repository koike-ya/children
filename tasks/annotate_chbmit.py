from datetime import datetime as dt
from pathlib import Path
import os

import pandas as pd
import numpy as np
import pyedflib
from tqdm import tqdm
from joblib import Parallel, delayed

from eeglibrary.src import EEG

import argparse

LABEL_COLUMNS = ['id', 'number', 'initial', 'date', 'start_time', 'end_time', 'abstruct', 'detail', 'label']
LABEL_KIND = ['none', 'seiz']
PHASES = ['train', 'val', 'test']
CHANNELS = ['Fp1', 'Fp2', 'O1', 'O2']
BASE_CHANNELS = ['A1', 'A2']


def annotate_args(parser):
    annotate_parser = parser.add_argument_group('annotation arguments')
    annotate_parser.add_argument('--include-artifact', action='store_true', help='Weather to include artifact or not')
    annotate_parser.add_argument('--n-jobs', type=int, default=4, help='Number of CPUs to use to annotate')
    annotate_parser.add_argument('--train-size', type=float, default=0.6, help='Train size')
    annotate_parser.add_argument('--val-size', type=float, default=0.2, help='Validation size')

    return parser


def load_edf(edf_path):
    edfreader = pyedflib.EdfReader(edf_path)
    return EEG.from_edf(edfreader, verbose=False)


def annotate_chbmit(data_dir, annotate_conf):
    """
    CHB-MITデータ・セットのアノテーションを行う
    各患者毎にsummaryからラベル情報を、edfファイルから脳波データを読み込んでいき、window秒のpklに変換して保存する。
    ラベルが途中で変わる区間は、保存はするがマニフェストファイルに記載しないことでデータに含めないようにする。

    データを読み込んで分割し、一旦保存する。次にラベルを解析してインデックスを計算し、保存したファイル名を変更することでアノテーションする
    chb04は途中でchannelが変更されているので除外
    """
    window_size = 10

    for patient_folder in Path(data_dir).iterdir():
        if not (patient_folder.is_dir() and patient_folder.name.startswith('chb')):
            continue

        with open(str(patient_folder / f'{patient_folder.name}-summary.txt'), 'r') as f:
            summary = f.read().split('\n\n')[2:]
        use_path_list = []

        # データの形式に問題があるため使用しない
        if patient_folder.name in ['chb04', 'chb09', 'chb11', 'chb12', 'chb13', 'chb15', 'chb16', 'chb17', 'chb18', 'chb19', 'chb24']:
            continue

        nth_edf = -1
        edf_path_list = [str(path).replace('+', '__') for path in patient_folder.iterdir() if path.suffix == '.edf']
        edf_path_list.sort()
        edf_path_list = [Path(path.replace('__', '+')) for path in edf_path_list]
        for edf_path in tqdm(edf_path_list, disable=False):
            nth_edf += 1

            save_dir = patient_folder / f'{edf_path.name[:-4]}'
            save_dir.mkdir(exist_ok=True)

            data = load_edf(str(edf_path))
            assert data.sr == 256
            assert data.sr * data.len_sec == data.values.shape[1]
            assert len(data.channel_list) == data.values.shape[0]

            # 先に保存してメモリエラーを回避して、ファイル名にだけ操作を加えてアノテーションする
            saved_list = data.split_and_save(window_size=window_size, n_jobs=annotate_conf['n_jobs'], padding=0,
                                             suffix='_none')
            del data
            print(summary[nth_edf])
            try:
                n_seizures = int(summary[nth_edf].split('\n')[3].split(': ')[-1])
            except ValueError as e:
                print(e)
                print(patient_folder.name, 'passed')
                continue

            remove_idx_list = []    # ラベルがまたがっているpklファイルの、saved_list内のindexを入れる
            for nth_seizure in range(n_seizures):
                start_sec = int(summary[nth_edf].split('\n')[4 + nth_seizure * 2].split(': ')[-1].replace(' ', '').replace('seconds', ''))
                end_sec = int(summary[nth_edf].split('\n')[5 + nth_seizure * 2].split(': ')[-1].replace(' ', '').replace('seconds', ''))

                # start_secやend_secがwindow_sizeで割って余るとき、その区間はラベルがまたがっている
                for sec in [start_sec, end_sec]:
                    if sec % window_size != 0:
                        remove_idx_list.append(sec // window_size)

                # seizureの区間のファイル名を変更する。
                # ラベルがまたがる区間はあとのマニフェスト作成時に抜かれるため、seizでラベル付されているが問題ない
                # end_secは最小単位が1なので、0.1引いてintにしてwindow_sizeで割ると、window_sizeの倍数のときだけ一つ繰り下げられる
                for index in range(start_sec // window_size, int(end_sec - 0.1) // window_size + 1):
                    path = Path(saved_list[index])
                    os.rename(path, str(path.parent / path.name.replace('none', 'seiz')))
                    saved_list[index] = str(path.parent / path.name.replace('none', 'seiz'))

            # indexが大きい順にsaved_listからパスを削除する
            remove_idx_list.sort(reverse=True)
            for index in remove_idx_list:
                saved_list.pop(index)

            use_path_list.extend(saved_list)

        pd.DataFrame(use_path_list).to_csv(save_dir.parent / 'manifest.csv', index=False, header=None)


if __name__ == '__main__':
    data_dir = '/media/cs11/Storage/koike/tmp/chb-mit-scalp-eeg-database-1.0.0/'
    parser = argparse.ArgumentParser(description='Annotation arguments')
    annotate_conf = vars(annotate_args(parser).parse_args())
    annotate_chbmit(data_dir, annotate_conf)
