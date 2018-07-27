import argparse
import openml
import os
import random
import sklearnbot
import uuid


def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    all_classifiers = ['decision_tree']
    parser.add_argument('--n_executions', type=int,  default=1000, help='number of runs to be executed. ')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
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
    openml.config.apikey = args.openml_apikey
    if args.openml_server:
        openml.config.server = args.openml_server
    tasks = openml.study.get_study(args.study_id, 'tasks').tasks

    configuration_space = sklearnbot.config_spaces.get_config_space(args.classifier, None)

    for i in range(args.n_executions):
        # get task and print meta-data
        task_id = random.choice(tasks)
        task = openml.tasks.get_task(task_id)
        data_name = task.get_dataset().name
        data_qualities = task.get_dataset().qualities
        data_tuple = (task.task_id, data_name, data_qualities['NumberOfFeatures'], data_qualities['NumberOfInstances'])
        print(sklearnbot.utils.get_time(), "Obtained task %d (%s); %s attributes; %s observations" % data_tuple)

        nominal_indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])
        numeric_indices = task.get_dataset().get_features_by_type('numeric', [task.target_name])
        classifier = sklearnbot.sklearn.deserialize(configuration_space, numeric_indices, nominal_indices)
        print(sklearnbot.utils.get_time(), classifier)

        # invoke OpenML run
        run = openml.runs.run_model_on_task(task, classifier)
        run.to_filesystem(os.path.join(args.output_dir, str(uuid.uuid4())), store_model=False)


if __name__ == '__main__':
    run()
