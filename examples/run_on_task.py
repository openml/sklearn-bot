import argparse
import openml
import os
import sklearnbot


# sshfs fr_jv1031@login1.nemo.uni-freiburg.de:/home/fr/fr_fr/fr_jv1031/experiments ~/nemo_experiments
def parse_args():
    all_classifiers = sklearnbot.config_spaces.get_available_config_spaces()
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    all_classifiers = ['decision_tree']
    parser.add_argument('--n_executions', type=int,  default=1000, help='number of runs to be executed. ')
    parser.add_argument('--task_id', type=int, required=True, help='the openml task id')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, default=None, help='the apikey to authenticate to OpenML')
    parser.add_argument('--classifier_name', type=str, choices=all_classifiers, default='decision_tree',
                        help='the classifier to run')
    default_output_dir = os.path.join(os.path.expanduser('~'), 'experiments/sklearn-bot')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                        help='Location to store finished runs')
    parser.add_argument('--upload_result', action='store_true',
                        help='if true, results will be immediately uploaded to OpenML.'
                             'Otherwise they will be stored on disk. ')

    return parser.parse_args()


def run():
    args = parse_args()
    if args.openml_apikey:
        openml.config.apikey = args.openml_apikey
    if args.openml_server:
        openml.config.server = args.openml_server
    else:
        openml.config.server = 'https://test.openml.org/api/v1/'

    configuration_space = sklearnbot.config_spaces.get_config_space(args.classifier_name, None)
    output_dir = os.path.join(args.output_dir, args.classifier_name)

    for i in range(args.n_executions):
        success, run_id, folder = sklearnbot.bot.run_bot_on_task(args.task_id,
                                                                 configuration_space,
                                                                 output_dir,
                                                                 args.upload_result)
        if success:
            print(sklearnbot.utils.get_time(), 'Run was executed successfully. Run id=%s; folder=%s' % (run_id, folder))
        else:
            print(sklearnbot.utils.get_time(), 'A problem occurred. Run id=%s; folder=%s' % (run_id, folder))


if __name__ == '__main__':
    run()
