import argparse
import logging
import openml
import os
import sklearnbot


# sshfs fr_jv1031@login1.nemo.uni-freiburg.de:/home/fr/fr_fr/fr_jv1031/experiments ~/nemo_experiments
def parse_args():
    all_classifiers = sklearnbot.config_spaces.get_available_config_spaces(True)
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--task_id', type=int, default=3, help='the openml task id')
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
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    args = parse_args()
    if args.openml_apikey:
        openml.config.apikey = args.openml_apikey
    if args.openml_server:
        openml.config.server = args.openml_server
    else:
        openml.config.server = 'https://test.openml.org/api/v1/'

    configuration_space_wrapper = sklearnbot.config_spaces.get_config_space(args.classifier_name, None)
    configuration_space_wrapper.wrap_in_fixed_pipeline()

    output_dir = os.path.join(args.output_dir, args.classifier_name)

    success, run_id, folder = sklearnbot.bot.run_optimizer_on_task(args.task_id,
                                                                   configuration_space_wrapper,
                                                                   output_dir,
                                                                   args.upload_result)
    if success:
        logging.info('Run was executed successfully. Run id=%s; folder=%s' % (run_id, folder))
    else:
        logging.warning('A problem occurred. Run id=%s; folder=%s' % (run_id, folder))


if __name__ == '__main__':
    run()
