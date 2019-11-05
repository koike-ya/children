from datetime import datetime as dt
from pathlib import Path
import os

import pandas as pd
import numpy as np
import pyedflib
from tqdm import tqdm

from eeglibrary.src import EEG

import argparse

LABEL_COLUMNS = ['id', 'number', 'initial', 'date', 'start_time', 'end_time', 'abstruct', 'detail', 'label']
LABEL_KIND = ['none', 'seiz', 'arti']
PHASES = ['train', 'val', 'test']
CHANNELS = ['Fp1', 'Fp2', 'O1', 'O2']
BASE_CHANNELS = ['A1', 'A2']


def annotate_args(parser):
    annotate_parser = parser.add_argument_group('annotation arguments')
    annotate_parser.add_argument('--include-artifact', action='store_true', help='Weather to include artiifact or not')
    annotate_parser.add_argument('--n-jobs', type=int, default=4, help='Number of CPUs to use to annotate')
    annotate_parser.add_argument('--train-size', type=float, default=0.6, help='Train size')
    annotate_parser.add_argument('--val-size', type=float, default=0.2, help='Validation size')

    return parser


def check_input_excel(excel_path):
    # ラベル情報を読み込み、ラベル情報の形式が正しいかをチェックする
    label_info = pd.read_excel(excel_path)

    if label_info.shape[1] == 9:
        label_info.columns = LABEL_COLUMNS
    else:
        print(f'Input excel shape should be (n, 9), but given excel has {label_info.shape}')
        exit()

    label_info = label_info[~pd.isna(label_info['id'])]
    return label_info


def load_edf(edf_path):
    edfreader = pyedflib.EdfReader(edf_path)
    return EEG.from_edf(edfreader)


def annotate(sr, label_info, with_artifact=False):
    start_time = label_info['start_time']
    label_list = []

    def time_to_index(time_):
        time_ = dt.strptime(time_, '%H:%M:%S').time()
        diff = {d: getattr(time_, d) - getattr(start_time, d) for d in ['hour', 'minute', 'second']}
        if diff['hour'] < 0:
            diff['hour'] += 24
        return (diff['hour'] * 60 * 60 + diff['minute'] * 60 + diff['second']) * sr

    labels = label_info['label'].split('\n')
    for label_time in labels:
        if label_time[:2] == '発作':
            label = 'seiz'
            duration = label_time[3:].split(',')
        elif label_time[:7] == 'アーチファクト':
            if not with_artifact:
                continue
            label = 'arti'
            duration = label_time[8:].split(',')
        else:
            raise NotImplementedError

        for d in duration:
            # 終わりが記載されていない場合は1秒間のラベルとしている
            start, end = d.split('-')
            s_index, e_index = time_to_index(start), time_to_index(end)
            label_list.append({'label': label, 's_index': s_index, 'e_index': e_index})

    label_list.sort(key=lambda x: x['s_index'])
    return label_list


def make_manifest(patient_id, renamed_list, train_size: float, val_size: float):
    size_list = [train_size, train_size + val_size, 1.0]
    save_dir = Path(renamed_list['none'][0]).parents[1]
    # データが保存されたパスのリストを受け取り、train, val, testに分割し、それぞれcsvファイルに保存する
    start_idx_list = {label: 0 for label in renamed_list.keys()}
    for phase, size in zip(PHASES, size_list):
        df = pd.DataFrame()

        for label in renamed_list.keys():
            df = pd.concat([df, pd.DataFrame(renamed_list[label][start_idx_list[label]:int(len(renamed_list[label]) * size)])])
            start_idx_list[label] = int(len(renamed_list[label]) * size)
        df.to_csv(save_dir / '{}_{}_manifest.csv'.format(patient_id, phase), index=False, header=None)


def make_edf_summary(excel_path):
    sr = 500
    label_info = check_input_excel(excel_path)
    for i, pat_info in label_info.iterrows():
        label_list = annotate(sr, pat_info, True)
        label_list = [(dic['s_index'], dic['e_index']) for dic in label_list if dic['label'] == 'seiz']

        if len(label_list) == 0:
            continue

        summary = ''
        summary += f"File Name: {pat_info['id']}_1-1.edf\n"
        summary += f"File Start Time: {pat_info['start_time']}\n"
        summary += f"File End Time: {pat_info['end_time']}\n"
        summary += f"Number of Seizures in File: {len(label_list)}"
        for i, (start, end) in enumerate(label_list):
            summary += f'\nSeizure {i+1} Start Time: {int(start // sr)} seconds'
            summary += f'\nSeizure {i+1} End Time: {int(end // sr)} seconds'

        with open(Path(excel_path).parent / f"{pat_info['id']}-summary.txt", 'w') as f:
            f.write(summary)


def annotate_child(excel_path, annotate_conf):
    """
    県立子ども病院のデータ・セットのアノテーションを行う
    inputは発作の時間とアーチファクトの時間が記載されたexcelファイル
    outputは、患者ID名のフォルダの中に、10秒ごとに分割したpklデータファイルと、そのファイル名にラベルがseizかnoneかがつけられたもの
    そしてマニフェストファイルを作成し保存する

    データを読み込んで分割し、一旦保存する。次にラベルを解析してインデックスを計算し、保存したファイル名を変更することでアノテーションする
    """
    data_dir = Path(excel_path).parent
    window_size = 16
    label_info = check_input_excel(excel_path)
    file_suffix = '_1-1.edf'

    for i, pat_info in label_info.iterrows():

        (data_dir / pat_info['id']).mkdir(exist_ok=True)
        # if list(Path(data_dir / pat_info['id']).iterdir()):
        #     continue

        data = load_edf(f"{data_dir}/{pat_info['id']}{file_suffix}")
        sr = data.sr

        signals = np.zeros((len(CHANNELS), data.values.shape[1]))
        channel_list = []
        for i, channel in enumerate(CHANNELS):
            signals[i] = data.values[data.channel_list.index(channel)] - data.values[data.channel_list.index(BASE_CHANNELS[i % 2]), :]
            channel_list.append(f'{channel}-{BASE_CHANNELS[i % 2]}')
        data.values = signals
        data.channel_list = channel_list
        splitted_data = data.split(window_size=window_size, n_jobs=annotate_conf['n_jobs'], padding=0)

        del data

        # 先に保存してメモリエラーを回避して、ファイル名にだけ操作を加えてアノテーションする
        saved_list = []
        labels = ['none', 'seiz', 'arti'] if annotate_conf['include_artifact'] else ['none', 'seiz']
        renamed_list = {label: [] for label in labels}

        for i, splitted in tqdm(enumerate(splitted_data), total=len(splitted_data)):
            s_index = i * sr * window_size
            file_name = f'{s_index}_{s_index + sr * window_size}.pkl'
            splitted.to_pkl(data_dir / pat_info['id'] / file_name)
            saved_list.append(data_dir / pat_info['id'] / file_name)

        label_list = annotate(sr, pat_info, annotate_conf['include_artifact'])

        label_pointer = 0   # ラベルの区間のうち、どこまで終了したかを示す。最大でlen(label_list)-1まで
        for i, saved_path in enumerate(saved_list):
            s_index = i * sr * window_size

            if label_pointer < len(label_list) and s_index > label_list[label_pointer]['e_index']:
                label_pointer += 1

            if label_pointer >= len(label_list):
                label = 'none'
            elif label_list[label_pointer]['s_index'] <= s_index < label_list[label_pointer]['e_index']:
                label = label_list[label_pointer]['label']
            else:
                label = 'none'

            os.rename(saved_path, f'{saved_path.parent}/{saved_path.stem}_{label}.pkl')
            renamed_list[label].append(f'{saved_path.parent}/{saved_path.stem}_{label}.pkl')

        del splitted_data

        make_manifest(pat_info['id'], renamed_list, annotate_conf['train_size'], annotate_conf['val_size'])


if __name__ == '__main__':
    excel_path = '/media/tomoya/SSD-PGU3/research/brain/children/eeg_annotation.xlsx'
    parser = argparse.ArgumentParser(description='Annotation arguments')
    annotate_conf = vars(annotate_args(parser).parse_args())
    annotate_child(excel_path, annotate_conf)
    # make_edf_summary(excel_path)