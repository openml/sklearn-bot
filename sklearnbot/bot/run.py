import logging
import openml
import os
import shutil
import sklearn
import sklearnbot
import traceback
import typing
import uuid

from sklearnbot.config_spaces import ConfigSpaceWrapper


def run_bot_on_task(task_id: int,
                    configuration_space_wrapper: ConfigSpaceWrapper,
                    run_defaults: bool,
                    output_dir: str,
                    upload_and_delete: bool) \
        -> typing.Tuple[bool, typing.Optional[int], typing.Optional[str]]:
    """
    Runs the bot with a random configuration on an OpenML task

    Parameters
    ----------
    task_id: int
        The OpenML task id to run the bot on

    run_defaults: bool
        If set to true, the configuration will be run with its default hyperparameter values

    configuration_space_wrapper: ConfigSpace.ConfigurationSpace
        The config space wrapper, that can be assembled to a config space, from which random configurations will be
        sampled

    output_dir: str
        A writable directory where the intermediate run results can be stored,
        before uploading

    upload_and_delete: bool
        If true, after the run has been executed it will be uploaded to OpenML.
        If the uploading is correct, the local files will be deleted afterwards.

    Returns
    -------
    success: bool
        A boolean indicating whether the operation (and and/or upload) was
        successful

    run_id: int or None
        If uploaded, the OpenML run id that was assigned to the run. None
        otherwise

    local_run_folder: str or None
        If the run was executed successfully and the folder was not deleted,
        the path to the folder. None otherwise
    """
    configuration_space = configuration_space_wrapper.assemble()
    local_run_dir = None
    try:
        # obtain task
        task = openml.tasks.get_task(task_id)
        data_name = task.get_dataset().name
        data_qualities = task.get_dataset().qualities
        data_tuple = (task.task_id, data_name, data_qualities['NumberOfFeatures'], data_qualities['NumberOfInstances'])
        logging.info('Obtained task %d (%s); %s attributes; %s observations' % data_tuple)

        # obtain deserialized classifier
        nominal_indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])
        numeric_indices = task.get_dataset().get_features_by_type('numeric', [task.target_name])
        if configuration_space_wrapper.wrapped_in_pipeline:
            classifier = sklearnbot.sklearn.as_pipeline(configuration_space, numeric_indices, nominal_indices)
        else:
            classifier = sklearnbot.sklearn.as_estimator(configuration_space, False)

        # sample configuration and set hyperparameters
        if not run_defaults:
            configuration = configuration_space.sample_configuration(1)
            logging.info('Configuration: %s' % configuration.get_dictionary())
            classifier.set_params(**configuration.get_dictionary())
        else:
            logging.info('Running default configuration')

        # invoke OpenML run
        run = openml.runs.run_model_on_task(classifier, task)
        score = run.get_metric_fn(sklearn.metrics.accuracy_score)
        logging.info('Task %d - %s; Accuracy: %0.2f' % (task_id, task.get_dataset().name, score.mean()))
        local_run_dir = os.path.join(output_dir, str(task_id), str(uuid.uuid4()))
        run.to_filesystem(local_run_dir, store_model=False)
        if upload_and_delete:
            run = run.publish()
            shutil.rmtree(local_run_dir)
            local_run_dir = None
        return True, run.run_id, local_run_dir
    except openml.exceptions.OpenMLServerException:
        traceback.print_exc()
        return False, None, local_run_dir
    except openml.exceptions.OpenMLRunsExistError:
        traceback.print_exc()
        return False, None, local_run_dir
