import argparse
from pathlib import Path

import pandas as pd


def aggregate_args(parser):
    agg_parser = parser.add_argument_group('aggregate arguments')
    agg_parser.add_argument('--expt-ids')
    agg_parser.add_argument('--expt-name')

    return parser


def aggregate(cfg):
    total = len(cfg['expt_ids'].split(','))

    for phase in ['val', 'test']:
        for i, expt_id in enumerate(cfg['expt_ids'].split(',')):
            if i == 0:
                df = pd.read_csv(Path(__file__).resolve().parent.parent / 'output' / 'metrics' / f"{expt_id}_{phase}.csv",
                                 index_col=0) / total
                continue

            df += pd.read_csv(Path(__file__).resolve().parent.parent / 'output' / 'metrics' / f"{expt_id}_{phase}.csv",
                              index_col=0) / total
        df.to_csv(Path(__file__).resolve().parent.parent / 'output' / 'metrics' / f"{cfg['expt_name']}_{phase}.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aggregate arguments')
    cfg = vars(aggregate_args(parser).parse_args())
    aggregate(cfg)
