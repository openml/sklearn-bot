# Scikit-learn bot
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Scikit-learn bot that can be used to automatically run scikit-learn classifiers
on OpenML tasks. This is an improved version of the bot that was created for the
[Parameter IMPortance](https://github.com/janvanrijn/openml-pimp) project. 

## Usage

To run the bot on a single task from OpenML, please use the following command:

```
python examples/run_on_task.py --task_id 5 --openml_server https://www.openml.org/api/v1/ --openml_apikey abcdef --classifier_name decision_tree --upload_result
```

Note the following command line options: 
* `n_executions`: By default, the bot will execute 1000 runs and terminate after
these. Using this option this behavior can be overridden to any other number of
runs. 
* `task_id`: the OpenML task id to run on. 
* `openml_server`: due to the beta-state of the sklearn bot, the default 
behavior is to upload results to the test server. By using this option, this 
behavior can be overridden
* `openml_apikey`: API key to authenticate yourself with. Can be found on your
OpenML profile. 
* `classifier_name`: the classifier to run. Currently, `decision_tree` is the
only legal option, but more will follow soon. 
* `output_dir`: local directory where the results can be stored before
uploading. 
* `upload_result`: the default behavior of the bot is to store the runs locally,
before uploading them to the server. By specifying this flag, the runs will be
uploaded and the local files will be deleted.

Additionally, the bot can be ran on a OpenML benchmark suite, for example the 
[OpenML100](https://arxiv.org/abs/1708.03731). The bot will execute a set 
of `n_executions`, each time selecting a task at random from the full set of 
tasks. 


To run the bot on a a benchmark suite from OpenML, please use the following
command:

```
python examples/run_on_task.py --study_id OpenML100 --openml_server https://www.openml.org/api/v1/ --openml_apikey abcdef --classifier_name decision_tree --upload_result
```

This function has the same command line options as `run_on_task`, except for the
option `task_id`. Additionally,
* `study_id` (int or string) refers to the study ID on which the bot shall be
ran. 

## Feature Requests

The following features will gradually be added to the sklearn-bot (contributors
are welcome):
* More classifiers. We want to extend the capabilities of the bot to all 
configuration spaces in auto-sklearn. 
* Non-static pipelines. Although the current definition of a pipeline is fixed, 
we aim to add the notion of non-static pipelines. This should be incorporated
in a modular way. 
* More sampling methods. Currently, the sklearn-bot sampled uniformly from a set
of tasks, however it would be great if it was able to sample according to the
number of runs per task present on OpenML. 

