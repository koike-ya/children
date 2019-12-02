import argparse
import os
from datetime import timedelta, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from eeglibrary.src import EEG

LABEL_COLUMNS = ['id', 'number', 'initial', 'date', 'start_time', 'end_time', 'abstruct', 'detail', 'label']
LABEL_KIND = ['none', 'seiz', 'arti']
PHASES = ['train', 'val', 'test']
CHANNELS = ['Fp1', 'Fp2', 'O1', 'O2']
BASE_CHANNELS = ['A1', 'A2']


PREICTAL_RANGE = 10
SR = 500    # sample rate
POST_ICTAL_RANGE = 10


def annotate_args(parser):
    annotate_parser = parser.add_argument_group('annotation arguments')
    annotate_parser.add_argument('--include-artifact', action='store_true', help='Weather to include artiifact or not')
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
    label_info['start_datetime'] = pd.to_datetime(label_info['date'].astype(str).str[:-8] + label_info['start_time'].astype(str))
    label_info['end_datetime'] = pd.to_datetime(label_info['date'].astype(str).str[:-8] + label_info['end_time'].astype(str))

    return label_info


def load_edf(edf_path):
    edfreader = pyedflib.EdfReader(edf_path)
    return EEG.from_edf(edfreader)


def annotate(label_info):
    ictal_list = []
    arti_list = []

    labels = label_info['label'].split('\n')
    for label_time in labels:
        if label_time[:2] == '発作':
            label = 'ictal'
            duration = label_time[3:].split(',')
        elif label_time[:7] == 'アーチファクト':
            label = 'arti'
            duration = label_time[8:].split(',')
        else:
            raise NotImplementedError

        tmp_date = label_info['start_datetime']
        for d in duration:
            # 終わりが記載されていない場合は1秒間のラベルとしている
            start, end = dt.strptime(d.split('-')[0], '%H:%M:%S'), dt.strptime(d.split('-')[1], '%H:%M:%S')
            start = tmp_date.replace(hour=start.hour, minute=start.minute, second=start.second)
            end = tmp_date.replace(hour=end.hour, minute=end.minute, second=end.second)

            if end < tmp_date:   # endのみ日付をまたいだとき
                end += timedelta(days=1)
            if start < tmp_date:  # startもendも日付をまたいだとき
                start += timedelta(days=1)

            tmp_date = end

            locals()[f'{label}_list'].append({'label': label, 'start': start, 'end': end})

    ictal_list.sort(key=lambda x: x['start'])
    return ictal_list, arti_list


def calc_other_label_section(pat_into, label_list, arti_list, sr=500):
    whole_edf_start = pat_into['start_datetime']
    whole_edf_end = pat_into['end_datetime']

    ictal_list = []
    preictal_list = []
    postictal_list = []
    interictal_list = []
    to_idx = to_idx_func(whole_edf_start, sr)

    for label in label_list + arti_list:
        label['s_index'] = to_idx(label['start'])
        label['e_index'] = to_idx(label['end'])

        if label['label'] == 'ictal':
            ictal_list.append(label)

    # preictal のラベル計算
    for i, ictal_section in enumerate(ictal_list):
        prev_ictal_section = {'end': whole_edf_start} if i == 0 else label_list[i - 1]

        preictal_start = max(prev_ictal_section['end'], ictal_section['start'] - timedelta(minutes=PREICTAL_RANGE))

        if preictal_start < ictal_section['start']:
            preictal_list.append({'label': 'pre', 'start': preictal_start, 'end': ictal_section['start'],
                                  's_index': to_idx(preictal_start), 'e_index': to_idx(ictal_section['start'])})

    # postictal のラベル計算
    diff = 1 if len(preictal_list) == len(ictal_list) else 0
    for i, ictal_section in enumerate(ictal_list):

        if i == len(ictal_list) - 1:
            next_preictal_start = whole_edf_end
        else:
            next_preictal_start = preictal_list[i + diff]['start']

        postictal_end = min(next_preictal_start, ictal_section['end'] + timedelta(minutes=POST_ICTAL_RANGE))
        if postictal_end > ictal_section['end']:

            postictal_list.append({'label': 'post', 'start': ictal_section['end'], 'end': postictal_end,
                                   's_index': to_idx(ictal_section['end']), 'e_index': to_idx(postictal_end)})

    ictal_list.extend(preictal_list)
    ictal_list.extend(postictal_list)
    ictal_list.sort(key=lambda x: x['s_index'])

    # interictalで埋める
    for i, section in enumerate(ictal_list):
        section_start = section['start']
        if i == 0:
            if section_start == whole_edf_start:
                continue
            else:
                prev_end = whole_edf_start
        else:
            prev_end = ictal_list[i - 1]['end']

        if i == len(ictal_list) - 1:
            if prev_end == whole_edf_end:
                continue
            else:
                section_start = whole_edf_end

        if prev_end < section_start:
            interictal_list.append({'label': 'inte', 'start': prev_end, 'end': section_start,
                                   's_index': to_idx(prev_end), 'e_index': to_idx(section_start)})

    ictal_list.extend(interictal_list)
    for section in ictal_list:
        assert section['end'] > section['start']
    ictal_list.sort(key=lambda x: x['s_index'])

    return ictal_list, arti_list


def to_idx_func(start_time, sr):
    def datetime_to_index(time_):
        assert start_time <= time_
        return (time_ - start_time).seconds * sr

    return datetime_to_index


def annotate_child(excel_path, annotate_conf):
    """
    県立子ども病院のデータ・セットのアノテーションを行う
    inputは発作の時間とアーチファクトの時間が記載されたexcelファイル
    outputは、患者ID名のフォルダの中に、10秒ごとに分割したpklデータファイルと、そのファイル名にラベルがseizかnoneかがつけられたもの
    そしてマニフェストファイルを作成し保存する

    データを読み込んで分割し、一旦保存する。次にラベルを解析してインデックスを計算し、保存したファイル名を変更することでアノテーションする
    """
    data_dir = Path(excel_path).parent
    window_size = 30
    window_stride = 15
    sr = 500
    label_info = check_input_excel(excel_path)
    file_suffix = '_1-1.edf'

    for i, pat_info in label_info.iterrows():

        (data_dir / pat_info['id']).mkdir(exist_ok=True)
        if pat_info['id'] == 'YJ01140M':
            continue

        if pat_info['start_datetime'] > pat_info['end_datetime']:
            pat_info['end_datetime'] += timedelta(days=1)

        label_list, arti_list = annotate(pat_info)

        if not label_list:
            print(f"{pat_info['id']} has no seizure section, so skipped.")
            continue

        label_list, arti_list = calc_other_label_section(pat_info, label_list, arti_list, sr)
        # print(label_list)

        # seiz_hour = 0
        # arti_hour = 0
        # none_seiz_hour = 0
        # print(pat_info['id'])
        # print(f'seiz_hour\tnone_seiz_hour\tartiafact_hour')
        #
        # for label_info in label_list:
        #     if label_info['label'] == 'ictal':
        #         seiz_hour += (label_info['end'] - label_info['start']).seconds / 3600
        #     elif label_info['label'] == 'arti':
        #         arti_hour += (label_info['end'] - label_info['start']).seconds / 3600
        #     elif label_info['label'] in ['inte', 'pre', 'post']:
        #         none_seiz_hour += (label_info['end'] - label_info['start']).seconds / 3600
        # print(f'{seiz_hour :.2f}\t{none_seiz_hour :.2f}\t{arti_hour :.2f}')
        # continue

        data = load_edf(f"{data_dir}/{pat_info['id']}{file_suffix}")

        signals = np.zeros((len(CHANNELS), data.values.shape[1]))
        channel_list = []

        if 'EEG' in data.channel_list[0]:
            data.channel_list = [c.replace('EEG ', '') for c in data.channel_list]
            data.channel_list = [c.replace('FP1', 'Fp1').replace('FP2', 'Fp2') for c in data.channel_list]

        for i, channel in enumerate(CHANNELS):
            signals[i] = data.values[data.channel_list.index(channel)] - data.values[data.channel_list.index(BASE_CHANNELS[i % 2]), :]
            channel_list.append(f'{channel}-{BASE_CHANNELS[i % 2]}')
        data.values = signals
        data.channel_list = channel_list
        saved_list = data.split_and_save(window_size=window_size, window_stride=window_stride,
                                         save_dir=data_dir / pat_info['id'], suffix='_none',
                                         n_jobs=annotate_conf['n_jobs'], padding=0)

        del data

        # 先に保存してメモリエラーを回避して、ファイル名にだけ操作を加えてアノテーションする

        renamed_list = []
        pointer = 0
        arti_pointer = 0    # artifactの区間を削除する

        for path in saved_list:
            if pointer >= len(label_list):
                continue

            path_s_idx, path_e_idx = list(map(int, Path(path).name.split('_')[:-1]))

            # pathが区間の内側にいるときのみラベルを変更する
            if path_s_idx >= label_list[pointer]['s_index'] and path_e_idx <= label_list[pointer]['e_index']:
                # pathがartifactの区間に触れているときは削除
                if arti_pointer < len(arti_list) and not path_e_idx <= arti_list[arti_pointer]['s_index']:
                    continue

                os.rename(path, f"{Path(path).parent}/{Path(path).name.replace('none', label_list[pointer]['label'])}")
                renamed_list.append(
                    f"{Path(path).parent}/{Path(path).name.replace('none', label_list[pointer]['label'])}")

            if path_e_idx >= label_list[pointer]['e_index']:
                pointer += 1

            if arti_pointer < len(arti_list) and path_s_idx >= arti_list[arti_pointer]['e_index']:
                arti_pointer += 1

        print(pd.Series(renamed_list).apply(
            lambda x: x.split('/')[-1].replace('.pkl', '').split('_')[-1]).value_counts())
        print(Path(renamed_list[0]).parents[1] / f"{pat_info['id']}_manifest.csv")
        pd.DataFrame(renamed_list).to_csv(Path(renamed_list[0]).parents[1] / f"{pat_info['id']}_manifest.csv",
                                          header=None, index=False)
        # exit()


if __name__ == '__main__':
    excel_path = '/home/tomoya/workspace/research/brain/children/input/eeg_annotation.xlsx'
    parser = argparse.ArgumentParser(description='Annotation arguments')
    annotate_conf = vars(annotate_args(parser).parse_args())
    annotate_child(excel_path, annotate_conf)
    # make_edf_summary(excel_path)
