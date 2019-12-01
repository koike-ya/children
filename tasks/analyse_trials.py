import numpy as np
import pandas as pd


def main():
    trials = pd.read_csv('trials_19111505.csv', index_col=0)
    trials.columns = list(trials.columns)[:5] + list(trials.iloc[0, 5:])
    trials = trials.iloc[1:, :]
    trials['value'] = trials['value'].fillna(10000)

    for phase in ['val', 'test']:
        for metric in ['accuracy', 'far', 'recall_1', 'loss']:
            trials[f'{phase}_{metric}'] = trials[f'{phase}_{metric}'].fillna('[0.0]').apply(lambda x: x[1:-1].replace('\n', '').split(' '))
            trials[f'{phase}_{metric}'] = trials[f'{phase}_{metric}'].apply(lambda x: [float(v) for v in x if v != ''])
            trials[f'{phase}_{metric}'] = trials[f'{phase}_{metric}'].apply(lambda x: np.array(x).mean())
            # trials[f'{phase}_{metric}_std'] = trials[f'{phase}_{metric}'].apply(lambda x: np.array(x).std())

    trials = trials.sort_values(by='value')
    a = ''


if __name__ == '__main__':
    main()