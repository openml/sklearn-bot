import arff
import argparse
import json
import openml
import openmlcontrib
import os
import pandas
import sklearnbot
import sklearn


def parse_args():
    all_classifiers = sklearnbot.config_spaces.get_available_config_spaces()
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--cache_directory', type=str, default=os.path.expanduser('~') + '/experiments/sklearn-bot/cache',
                        help='directory to store cache. Please delete this regularly, to obtain the latest results')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/sklearn-bot/results',
                        help='directory to store output')
    parser.add_argument('--num_runs', type=int, default=500, help='max results per task to obtain, to limit time')
    parser.add_argument('--study_id', type=str, default=14, help='the tag to obtain the tasks from')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy', help='the evaluation measure of interest')
    parser.add_argument('--normalize', action='store_true',
                        help='if true, scales the performance result per task to [0, 1]')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--classifier_name', type=str, choices=all_classifiers, default='decision_tree',
                        help='the classifier to run')
    return parser.parse_args()


def run():
    args = parse_args()
    if args.openml_server:
        openml.config.server = args.openml_server
    else:
        openml.config.server = 'https://test.openml.org/api/v1/'

    study = openml.study.get_study(args.study_id, 'tasks')
    setup_data_all = None
    scaler = sklearn.preprocessing.MinMaxScaler()

    # acquire config space
    config_space = sklearnbot.config_spaces.get_config_space(args.classifier_name, None)
    # acquite classifier and flow, for flow id
    clf = sklearnbot.sklearn.deserialize(config_space, [], [])
    flow = openml.flows.sklearn_to_flow(clf)
    flow_id = openml.flows.flow_exists(flow.name, flow.external_version)

    if flow_id is False:
        raise ValueError('Flow not recognized, this means that it does not exist on the OpenML server yet.')

    relevant_tasks = study.tasks
    # if args.num_tasks:
    relevant_tasks = study.tasks[:5]

    for task_id in relevant_tasks:
        print(sklearnbot.utils.get_time(), 'Currently processing task', task_id)
        try:
            setup_data = openmlcontrib.meta.get_task_flow_results_as_dataframe(task_id=task_id,
                                                                               flow_id=flow_id,
                                                                               num_runs=args.num_runs,
                                                                               raise_few_runs=False,
                                                                               configuration_space=config_space,
                                                                               parameter_field='name',
                                                                               evaluation_measure=args.scoring,
                                                                               cache_directory=args.cache_directory)
        except ValueError as e:
            print('Problem in task %d:' % task_id, e)
            continue
        setup_data['task_id'] = task_id
        if setup_data_all is None:
            setup_data_all = setup_data
        else:
            if list(setup_data.columns.values) != list(setup_data_all.columns.values):
                raise ValueError()
            if args.normalize:
                setup_data[['y']] = scaler.fit_transform(setup_data[['y']])

            setup_data_all = pandas.concat((setup_data_all, setup_data))

    # if len(setup_data_all) < args.num_runs * len(relevant_tasks) * 0.25:
    #     raise ValueError('Num results suspiciously low. Please check.')

    os.makedirs(args.output_directory, exist_ok=True)
    # create the task / parameters / performance arff
    filename = os.path.join(args.output_directory, 'results_%s.arff' % args.classifier_name)
    relation_name = 'openml-meta-flow-%d' % flow_id
    json_meta = {'flow_id': flow_id,
                 'openml_server': openml.config.server,
                 'measure': args.scoring,
                 'normalized_y': args.normalize,
                 'study_id': args.study_id,
                 'num_runs': args.num_runs}
    with open(filename, 'w') as fp:
        arff.dump(openmlcontrib.meta.dataframe_to_arff(setup_data_all,
                                                       relation_name,
                                                       json.dumps(json_meta)), fp)
    print(sklearnbot.utils.get_time(), 'Stored ARFF file with vanilla results to', filename)

    # # create the task meta-features * parameters * performance arff
    # task_qualities = {}
    # for task_id in relevant_tasks:
    #     task = openml.tasks.get_task(task_id)
    #     task_qualities[task_id] = task.get_dataset().qualities
    # # index of qualities: the task id
    # qualities_with_na = pandas.DataFrame.from_dict(task_qualities, orient='index', dtype=np.float)
    # qualities = pandas.DataFrame.dropna(qualities_with_na, axis=1, how='any')
    #
    # setup_data_with_meta_features = setup_data_all.join(qualities, on='task_id', how='inner')
    #
    # filename = os.path.join(args.output_directory, 'meta_%s.arff' % args.classifier_name)
    # with open(filename, 'w') as fp:
    #     arff.dump(openmlcontrib.meta.dataframe_to_arff(setup_data_with_meta_features,
    #                                                    relation_name,
    #                                                    json.dumps(json_meta)), fp)


if __name__ == '__main__':
    run()
