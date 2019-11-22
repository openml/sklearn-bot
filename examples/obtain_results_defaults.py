import argparse
import logging
import openml
import os
import sklearnbot
import random


# sshfs fr_jv1031@login1.nemo.uni-freiburg.de:/home/fr/fr_fr/fr_jv1031/experiments ~/nemo_experiments
def parse_args():
    all_classifiers = sklearnbot.config_spaces.get_available_config_spaces(True)
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--study_id', type=str, default='OpenML-CC18', help='the tag to obtain the tasks from')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, default=None, help='the apikey to authenticate to OpenML')
    parser.add_argument('--classifier_name', type=str, choices=all_classifiers, default='adaboost',
                        help='the classifier to run')
    parser.add_argument('--vanilla_estimator', action='store_true',
                        help='if set, run vanilla classifiers rather than a pipeline')
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
        openml.config.server = 'https://www.openml.org/api/v1/'
    tasks = openml.study.get_suite(args.study_id).tasks

    configuration_space_wrapper = sklearnbot.config_spaces.get_config_space(args.classifier_name, None)
    if not args.vanilla_estimator:
        configuration_space_wrapper.wrap_in_fixed_pipeline()

    setup_ids = set()
    for task_id in tasks:
        task = openml.tasks.get_task(task_id)
        classifier = sklearnbot.bot.prepare_classifier(configuration_space_wrapper, task, True)
        extension = openml.extensions.get_extension_by_model(classifier)
        flow = extension.model_to_flow(classifier)
        flow_id = openml.flows.flow_exists(flow.name, flow.external_version)

        if flow_id is False:
            logging.info('Task %d: Can not find flow' % task_id)
            continue
        server_flow = openml.flows.get_flow(flow_id)
        openml.flows.flow._copy_server_fields(server_flow, flow)
        try:
            setup_id = openml.setups.setup_exists(flow)
            if setup_id is False:
                logging.info('Task %d: Can not find setup' % task_id)
            else:
                logging.info('Task %d: Setup id %d' % (task_id, setup_id))
                setup_ids.add(setup_id)
                print(setup_ids)
        except openml.exceptions.OpenMLServerException:
            pass
    print(setup_ids)


if __name__ == '__main__':
    run()
