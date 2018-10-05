import ConfigSpace
import openml
import os
import random
import sklearnbot
import typing
import traceback
import uuid


def run_on_random_task(tasks: typing.List, configuration_space: ConfigSpace, output_dir: str):
    task_id = random.choice(tasks)
    try:
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
        run.to_filesystem(os.path.join(output_dir, str(task_id), str(uuid.uuid4())),
                          store_model=False)
    except openml.exceptions.OpenMLServerException:
        traceback.print_exc()
