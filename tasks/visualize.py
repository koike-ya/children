import argparse
import io
from pathlib import Path

import pandas as pd


def aggregate_args(parser):
    agg_parser = parser.add_argument_group('aggregate arguments')
    agg_parser.add_argument('--expt-ids')
    agg_parser.add_argument('--expt-name')

    return parser


def visualize(cfg):
    stat = 'mean'
    val_agg_df = pd.DataFrame()
    test_agg_df = pd.DataFrame()
    for expt_id in cfg['expt_ids'].split(','):
        for phase in ['val', 'test']:

            locals()[f'{phase}_metrics'] = pd.read_csv(Path(__file__).resolve().parent.parent / 'output' / 'metrics' / f"{expt_id}_{phase}.csv")

            series = pd.Series()
            for i, row in locals()[f'{phase}_metrics'].iterrows():
                for col in ['mean', 'std']:
                    series[f"{row['metric_name']}_{col}"] = row[col]
            _ = pd.DataFrame(series).T
            _.index = [expt_id]
            _ = _[[c for c in _.columns if stat in c]]
            _.columns = [c[:-5] for c in _.columns]

            if phase == 'val':
                val_agg_df = pd.concat([val_agg_df, _])
            else:
                test_agg_df = pd.concat([test_agg_df, _])

    test_agg_df.to_csv(Path(__file__).resolve().parent.parent / 'output' / 'metrics' / f"vis-test-{cfg['expt_name']}_{stat}.csv")
    val_agg_df.to_csv(Path(__file__).resolve().parent.parent / 'output' / 'metrics' / f"vis-val-{cfg['expt_name']}_{stat}.csv")

    import matplotlib.pyplot as plt
    test_agg_df.T.plot.bar(rot=0, figsize=(10, 6))
    plt.savefig(Path(__file__).resolve().parent.parent / 'output' / 'metrics' / f"test-{cfg['expt_name']}_{stat}.png")
    val_agg_df.T.plot.bar(rot=0, figsize=(10, 6))
    plt.savefig(Path(__file__).resolve().parent.parent / 'output' / 'metrics' / f"val-{cfg['expt_name']}_{stat}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aggregate arguments')
    cfg = vars(aggregate_args(parser).parse_args())
    visualize(cfg)
