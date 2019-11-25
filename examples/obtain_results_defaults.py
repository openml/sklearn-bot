import argparse
import joblib
import logging
import openml
import os
import pandas as pd
import sklearnbot
import typing


# sshfs fr_jv1031@login1.nemo.uni-freiburg.de:/home/fr/fr_fr/fr_jv1031/experiments ~/nemo_experiments
def parse_args():
    all_classifiers = sklearnbot.config_spaces.get_available_config_spaces(True)
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--study_id', type=str, default='OpenML-CC18', help='the tag to obtain the tasks from')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, default=None, help='the apikey to authenticate to OpenML')
    parser.add_argument('--classifier_name', type=str, choices=all_classifiers, default='adaboost',
                        help='the classifier to run')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~/experiments/sklearn-bot'))
    return parser.parse_args()


def get_setup_ids(tasks: typing.List[int], classifier_name: str) -> typing.Set[int]:
    configuration_space_wrapper = sklearnbot.config_spaces.get_config_space(classifier_name, None)
    configuration_space_wrapper.wrap_in_fixed_pipeline()

    setup_ids = set()
    for task_id in tasks:
        task = openml.tasks.get_task(task_id)
        classifier = sklearnbot.bot.prepare_classifier(configuration_space_wrapper, task, True)
        extension = openml.extensions.get_extension_by_model(classifier)
        flow = extension.model_to_flow(classifier)
        flow_id = openml.flows.flow_exists(flow.name, flow.external_version)

        if flow_id is False:
            logging.info('Task %d: Can not find flow for classifier %s' % (task_id, classifier_name))
            continue
        server_flow = openml.flows.get_flow(flow_id)
        openml.flows.flow._copy_server_fields(server_flow, flow)
        try:
            setup_id = openml.setups.setup_exists(flow)
            if setup_id is False:
                logging.info('Task %d: Can not find setup for classifier %s' % (task_id, classifier_name))
            else:
                logging.info('Task %d, classifier %s: Setup id %d' % (task_id, classifier_name, setup_id))
                setup_ids.add(setup_id)
        except openml.exceptions.OpenMLServerException:
            pass
    return setup_ids


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
    os.makedirs(args.output_directory, exist_ok=True)

    memory = joblib.memory.Memory(os.path.join(args.output_directory, '.cache'), verbose=0)
    get_setup_ids_cached = memory.cache(get_setup_ids)
    if args.classifier_name == 'all':
        classifiers = sklearnbot.config_spaces.get_available_config_spaces(False)
    else:
        classifiers = [args.classifier_name]

    results = pd.DataFrame()
    for classifier in classifiers:
        setup_ids = get_setup_ids_cached(tasks, classifier)
        logging.info('%s: %s' % (classifier, setup_ids))
        run_frame = openml.runs.list_runs(task=tasks, setup=list(setup_ids), output_format='dataframe')
        if len(results) == 0:
            results = run_frame
        else:
            results = results.append(run_frame)
    result_file = os.path.join(args.output_directory, 'run_results_%s.csv' % args.classifier_name)
    results.to_csv(result_file)
    logging.info('stored result to: %s' % result_file)


if __name__ == '__main__':
    run()
