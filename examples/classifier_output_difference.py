import argparse
import arff
import logging
import matplotlib.pyplot as plt
import numpy as np
import openml
import os
import pandas as pd
import scipy.cluster.hierarchy
import scipy.spatial.distance
import typing
import urllib.request


# runs after obtain_results_defaults
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default=os.path.expanduser('~/run_results_all.csv'))
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~/experiments/sklearn-bot'))
    parser.add_argument('--extension', type=str, default='pdf')
    return parser.parse_args()


def flow_name_neat(name):
    if name.strip()[-2:] == '))':
        name_splitted = name.split('(')
        return name_splitted[-2].split('.')[-1].split('(')[0] + '(' + name_splitted[-1].split('.')[-1] + ')'
    else:
        return name.split('.')[-1].replace(')', '')


def plot(df_results: np.array, labels: typing.List[str], output_file: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    scipy.cluster.hierarchy.dendrogram(
        scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(df_results), "single"),
        ax=ax, orientation='top', labels=np.array(labels))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file)
    logging.info('Saved to %s' % output_file)


def run():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    args = parse_args()

    df_runids = pd.read_csv(args.input_file)
    df_runids = df_runids.pivot(index='task_id', columns='flow_id', values='run_id')

    os.makedirs(os.path.join(args.output_directory, 'cod'), exist_ok=True)
    task_data = openml.tasks.list_tasks(task_id=df_runids.index)
    flow_data = openml.flows.list_flows()

    result = None
    labels = None

    for idx, (task_id, row) in enumerate(df_runids.iterrows()):
        task_n = task_data[task_id]['NumberOfInstances']
        logging.info('(%d/%d) Task %d: %d observations' % (idx+1, len(df_runids), task_id, task_n))
        result_list = list()
        for flow_id1, run_id1 in row.iteritems():
            result_list.append({'flow_id': flow_id1, 'B': flow_id1, 'cod': 0})
            for flow_id2, run_id2 in row.iteritems():
                if flow_id2 <= flow_id1:
                    continue
                path = '%s/%d_%d_%d.arff' % (os.path.join(args.output_directory, 'cod'), task_id, flow_id1, flow_id2)
                if not os.path.exists(path):
                    url = 'https://www.openml.org/api_splits/different_predictions/%d,%d' % (run_id1, run_id2)
                    urllib.request.urlretrieve(url, path)

                with open(path, 'r') as fp:
                    try:
                        data = arff.load(fp)
                        cod = len(data['data']) / task_n
                    except arff.BadDataFormat:
                        cod = 0

                result_list.append({'flow_id': flow_id1, 'B': flow_id2, 'cod': cod})
                result_list.append({'flow_id': flow_id2, 'B': flow_id1, 'cod': cod})
        df_results = pd.DataFrame(result_list)
        df_results = df_results.pivot(index='flow_id', columns='B', values='cod')
        df_results = df_results.reindex(sorted(df_results.columns), axis=1)
        labels_current = [flow_name_neat(flow_data[flow_id]['name']) for flow_id in df_results.columns.values]

        if result is None:
            result = df_results.values
            labels = labels_current
        else:
            result += df_results.values
            if labels != labels_current:
                raise ValueError()

        output_file = os.path.join(args.output_directory, 'task_%d.%s' % (task_id, args.extension))
        plot(df_results.values, labels, output_file)

    output_file = os.path.join(args.output_directory, 'all.%s' % args.extension)
    plot(result, labels, output_file)


if __name__ == '__main__':
    run()
