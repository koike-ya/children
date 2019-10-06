from datetime import datetime as dt
from pathlib import Path
import os

import pandas as pd
import pyedflib
from tqdm import tqdm

from eeglibrary.src import EEG

import argparse

LABEL_COLUMNS = ['id', 'number', 'initial', 'date', 'start_time', 'end_time', 'abstruct', 'detail', 'label']
LABEL_KIND = ['none', 'seiz', 'arti']
PHASES = ['train', 'val', 'test']


def annotate_args(parser):
    annotate_parser = parser.add_argument_group('annotation arguments')
    annotate_parser.add_argument('--include-artifact', action='store_true', help='Weather to include archifact or not')
    annotate_parser.add_argument('--n-jobs', type=int, default=4, help='Number of CPUs to use to annotate')

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
        return (diff['hour'] * 60 * 60 + diff['minute'] * 60 + diff['second']) * sr

    labels = label_info['label'].split('\n')
    for label_time in labels:
        if label_time[:2] == '発作':
            label = 'seiz'
            duration = label_time[3:].split(',')
        elif label_time[:7] == 'アーチファクト':
            if not with_artifact:
                continue
            label = 'arch'
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
    # if not Path(save_dir / patient_id).is_dir():
    #     return
    # path_list = list(Path(save_dir / patient_id).iterdir())
    # path_list.sort(key=lambda x: int(x.name.split('_')[0]))
    size_list = [train_size, train_size + val_size, 1.0]
    save_dir = Path(renamed_list[0]).parents[1]
    # データが保存されたパスのリストを受け取り、train, val, testに分割し、それぞれcsvファイルに保存する
    start_idx = 0
    for phase, size in zip(PHASES, size_list):
        pd.DataFrame(renamed_list[start_idx:int(len(renamed_list) * size)]).to_csv(
            save_dir / '{}_{}_manifest.csv'.format(patient_id, phase), index=False, header=None)
        start_idx = int(len(renamed_list) * size)


def annotate_child(excel_path, annotate_conf):
    """
    県立子ども病院のデータ・セットのアノテーションを行う
    inputは発作の時間とアーチファクトの時間が記載されたexcelファイル
    outputは、患者ID名のフォルダの中に、10秒ごとに分割したpklデータファイルと、そのファイル名にラベルがseizかnoneかがつけられたもの
    そしてマニフェストファイルを作成し保存する

    データを読み込んで分割し、一旦保存する。次にラベルを解析してインデックスを計算し、保存したファイル名を変更することでアノテーションする
    """
    data_dir = Path(excel_path).parent
    window_size = 1
    label_info = check_input_excel(excel_path)
    file_suffix = '_1-1.edf'

    for i, pat_info in label_info.iterrows():

        (data_dir / pat_info['id']).mkdir(exist_ok=True)
        # if list(Path(data_dir / pat_info['id']).iterdir()):
        #     continue

        data = load_edf(f"{data_dir}/{pat_info['id']}{file_suffix}")
        sr = data.sr
        electrodes = [2, 3, 6, 7]
        data.values = data.values[electrodes, :]
        data.channel_list = [data.channel_list[i] for i in electrodes]
        splitted_data = data.split(window_size=window_size, n_jobs=annotate_conf['n_jobs'], padding=0)

        del data

        # 先に保存してメモリエラーを回避して、ファイル名にだけ操作を加えてアノテーションする
        saved_list = []
        renamed_list = []
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
            renamed_list.append(f'{saved_path.parent}/{saved_path.stem}_{label}.pkl')

        del splitted_data

        make_manifest(pat_info['id'], renamed_list, annotate_conf['train_size'], annotate_conf['val_size'])


if __name__ == '__main__':
    excel_path = '/media/tomoya/SSD-PGU3/research/brain/children/eeg_annotation.xlsx'
    parser = argparse.ArgumentParser(description='Annotation arguments')
    annotate_conf = vars(annotate_args(parser).parse_args())
    annotate_child(excel_path, annotate_conf)
