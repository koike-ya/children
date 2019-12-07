import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from eeglibrary import eeg
from eeglibrary.src.eeg_dataloader import set_dataloader as eeg_dataloader
from eeglibrary.src.eeg_dataset import EEGDataSet
from eeglibrary.src.metrics import Metric
from eeglibrary.src.preprocessor import preprocess_args
from ml.src.dataloader import set_dataloader, set_ml_dataloader
from ml.src.metrics import metrics2df
from train_manager import TrainManager, train_manager_args


def aggregate_args(parser):
    agg_parser = parser.add_argument_group('aggregate arguments')
    agg_parser.add_argument('--expt-ids')
    agg_parser.add_argument('--expt-name')

    return parser


def aggregate(cfg):
    for phase in ['val', 'test']:
        df = pd.DataFrame()

        for expt_id in cfg['expt_ids'].split(','):
            locals()[f'{phase}_metrics'] = pd.read_csv(Path(__file__).parent.parent / 'output' / 'metrics' / f"{expt_id}_{phase}.csv")

            series = pd.Series()
            for i, row in locals()[f'{phase}_metrics'].iterrows():
                for col in ['mean', 'std']:
                    series[f"{row['metric_name']}_{col}"] = row[col] * i
            df = pd.concat([df, pd.DataFrame(series).T])

        df = pd.DataFrame(df.mean().values.reshape((-1, 2)))
        df.index = locals()[f'{phase}_metrics']['metric_name']
        df.columns = [f'{phase} mean mean', f'{phase} std mean']

        if phase == 'val':
            val_agg_df = df
        else:
            test_agg_df = df

    pd.concat([val_agg_df, test_agg_df], axis=1).to_csv(
        Path(__file__).parent.parent / 'output' / 'metrics' / f"agg-{cfg['expt_name']}.csv")
    # pd.concat([val_agg_df, test_agg_df], axis=1).plot.bar()
    # import matplotlib.pyplot as plt
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aggregate arguments')
    cfg = vars(aggregate_args(parser).parse_args())
    aggregate(cfg)
