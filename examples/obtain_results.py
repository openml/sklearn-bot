import arff
import argparse
import json
import logging
import numpy as np
import openml
import openmlcontrib
import os
import sklearnbot

IMPUTE_NA = -99999  # Placeholder for nan values


def parse_args():
    all_classifiers = sklearnbot.config_spaces.get_available_config_spaces(False)
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/sklearn-bot',
                        help='directory to store output')
    parser.add_argument('--extension', type=str, choices=['arff', 'csv'], default='csv')
    parser.add_argument('--num_runs', type=int, default=500, help='max results per task to obtain, to limit time')
    parser.add_argument('--study_id', type=str, default=14, help='the tag to obtain the tasks from')
    parser.add_argument('--scoring', type=str, nargs='+', default=['predictive_accuracy'],
                        help='the evaluation measure(s) of interest')
    parser.add_argument('--per_fold', action='store_true',
                        help='if true, obtains the results per fold (opposed to averaged results)')
    parser.add_argument('--normalize', action='store_true',
                        help='if true, scales the performance result per task to [0, 1]')
    parser.add_argument('--meta_features', action='store_true',
                        help='if true, also creates a dataset with the meta-features')
    parser.add_argument('--raise_few_runs', action='store_true', help='if true, it enforces that each task has the '
                                                                      'minimum amount of runs, or it raises an error')
    parser.add_argument('--openml_apikey', type=str, default=None, help='the openml api key')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--classifier_name', type=str, choices=all_classifiers, default='decision_tree',
                        help='the classifier to run')
    parser.add_argument('--vanilla_estimator', action='store_true',
                        help='if set, run vanilla classifiers rather than a pipeline')
    parser.add_argument('--flow_id', type=int, default=None,
                        help='if known, the flow id of the classifier (can be induced)')
    args_ = parser.parse_args()
    return args_


def run():
    args = parse_args()
    openml.config.apikey = args.openml_apikey
    if args.openml_server:
        openml.config.server = args.openml_server
    else:
        openml.config.server = 'https://test.openml.org/api/v1/'

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    logging.info('started obtain results script with parameters %s' % str(args))
    study = openml.study.get_suite(args.study_id)
    logging.info('obtained study %s with %d tasks' % (args.study_id, len(study.tasks)))

    # acquire config space
    configuration_space_wrapper = sklearnbot.config_spaces.get_config_space(args.classifier_name, None)
    if not args.vanilla_estimator:
        configuration_space_wrapper.wrap_in_fixed_pipeline()
    config_space = configuration_space_wrapper.assemble()
    # acquire classifier and flow, for flow id
    if configuration_space_wrapper.wrapped_in_pipeline:
        clf = sklearnbot.sklearn.as_pipeline(config_space, [], [])
    else:
        clf = sklearnbot.sklearn.as_estimator(config_space, False)

    if args.flow_id is None:
        openml_extension = openml.extensions.get_extension_by_model(clf)
        flow = openml_extension.model_to_flow(clf)
        flow_id = openml.flows.flow_exists(flow.name, flow.external_version)
    else:
        flow_id = args.flow_id
        flow = openml.flows.get_flow(flow_id)

    # check if hyperparameters are available
    for param_name in config_space.get_hyperparameter_names():
        parts = param_name.split("__")
        if len(parts) == 1:
            subflow = flow
        else:
            subflow = flow.get_subflow(parts[:-1])
        if parts[-1] not in subflow.parameters:
            raise ValueError('Missing hyperparameter %s in flow %s' % (param_name, flow.name))

    cache_directory = os.path.join(args.output_directory, 'cache')

    if flow_id is False:
        raise ValueError('Flow not recognized, this means that it does not exist on the OpenML server yet.')

    performance_data = openmlcontrib.meta.get_tasks_result_as_dataframe(
        task_ids=study.tasks,
        flow_id=flow_id,
        num_runs=args.num_runs,
        per_fold=args.per_fold,
        raise_few_runs=args.raise_few_runs,
        configuration_space=config_space,
        evaluation_measures=args.scoring,
        normalize=args.normalize,
        cache_directory=cache_directory
    )

    # if len(setup_data_all) < args.num_runs * len(relevant_tasks) * 0.25:
    #     raise ValueError('Num results suspiciously low. Please check.')

    output_directory = os.path.join(args.output_directory, 'results')
    os.makedirs(output_directory, exist_ok=True)
    # create the task / parameters / performance arff
    filename = os.path.join(output_directory,
                            'results__%d__%s__%s.%s' % (args.num_runs, args.classifier_name,
                                                        '__'.join(args.scoring), args.extension))
    relation_name = 'openml-meta-flow-%d' % flow_id
    json_meta = {
        'flow_id': flow_id,
        'openml_server': openml.config.server,
        'col_measures': args.scoring,
        'col_parameters': config_space.get_hyperparameter_names(),
        'normalized_y': args.normalize,
        'study_id': args.study_id,
        'max_runs_per_task': args.num_runs
    }
    with open(filename, 'w') as fp:
        if args.extension == 'arff':
            arff.dump(openmlcontrib.meta.dataframe_to_arff(performance_data,
                                                           relation_name,
                                                           json.dumps(json_meta)), fp)
        elif args.extension == 'csv':
            performance_data.to_csv(fp, index=False)
        else:
            raise ValueError()
    logging.info('Stored performance results as ARFF file with vanilla results to %s ' % filename)

    if args.meta_features:
        # create the task meta-features * parameters * performance arff
        meta_features = openmlcontrib.meta.get_tasks_qualities_as_dataframe(study.tasks, False, IMPUTE_NA, True)
        setup_data_with_meta_features = performance_data.join(meta_features, on='task_id', how='inner')

        filename = os.path.join(output_directory,
                                'metafeatures__%d__%s__%s.arff' % (args.num_runs, args.classifier_name,
                                                                   '__'.join(args.scoring)))
        with open(filename, 'w') as fp:
            if args.extension == 'arff':
                arff.dump(openmlcontrib.meta.dataframe_to_arff(setup_data_with_meta_features,
                                                               relation_name,
                                                               json.dumps(json_meta)), fp)
            elif args.extension == 'csv':
                setup_data_with_meta_features.to_csv(fp, index=False)
            else:
                raise ValueError()
        logging.info('Stored meta-features and results as ARFF file with vanilla results to %s' % filename)


if __name__ == '__main__':
    run()
