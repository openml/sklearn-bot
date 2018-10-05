import argparse
import openml
import os
import random
import sklearnbot
import traceback
import uuid


# sshfs fr_jv1031@login1.nemo.uni-freiburg.de:/home/fr/fr_fr/fr_jv1031/experiments ~/nemo_experiments
def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    all_classifiers = ['decision_tree']
    parser.add_argument('--n_executions', type=int,  default=1000, help='number of runs to be executed. ')
    parser.add_argument('--study_id', type=str, default=14, help='the tag to obtain the tasks from')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, default=None, help='the apikey to authenticate to OpenML')
    parser.add_argument('--classifier', type=str, choices=all_classifiers, default='decision_tree',
                        help='the classifier to run')
    default_output_dir = os.path.join(os.path.expanduser('~'), 'experiments/sklearnbot')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                        help='Location to store finished runs')
    return parser.parse_args()


def run():
    args = parse_args()
    if args.openml_apikey:
        openml.config.apikey = args.openml_apikey
    if args.openml_server:
        openml.config.server = args.openml_server
    else:
        openml.config.server = 'https://test.openml.org/api/v1/'
    tasks = openml.study.get_study(args.study_id, 'tasks').tasks

    configuration_space = sklearnbot.config_spaces.get_config_space(args.classifier, None)

    for i in range(args.n_executions):
        # get task and print meta-data
        sklearnbot.bot.run_on_random_task(tasks, configuration_space, args.output_dir)


if __name__ == '__main__':
    run()
