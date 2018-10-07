# sklearn-bot
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

sklearn-bot that can be used to automatically run scikit-learn classifiers
on OpenML tasks. This is an improved version of the sklearn-bot that was created
for the [Parameter IMPortance](https://github.com/janvanrijn/openml-pimp)
project. 

## Usage
Currently, two use cases are supported. Furthermore, we describe how to obtain
the results back from OpenML, once uploaded. 

### Single Task
To run the sklearn-bot on a single task from OpenML, please use the following
command:
```
python examples/run_on_task.py --task_id 5 --openml_server https://www.openml.org/api/v1/ --openml_apikey abcdef --classifier_name decision_tree --upload_result
```

The following command line options are accepted: 
* `n_executions`: By default, the sklearn-bot will execute 1000 runs and 
terminate after these. Using this option this behavior can be overridden to any
other number of runs. 
* `task_id`: the OpenML task id to run on. 
* `openml_server`: due to the beta-state of the sklearn-bot, the default 
behavior is to upload results to the test server. By using this option, this 
behavior can be overridden
* `openml_apikey`: API key to authenticate yourself with. Can be found on your
OpenML profile. 
* `classifier_name`: the classifier to run. Currently, `decision_tree` is the
only legal option, but more will follow soon. 
* `output_dir`: local directory where the results can be stored before
uploading. 
* `upload_result`: the default behavior of the sklearn-bot is to store the runs
locally, before uploading them to the server. By specifying this flag, the runs
will be uploaded and the local files will be deleted.

Additionally, the sklearn-bot can be ran on a OpenML benchmark suite, for
example the [OpenML100](https://arxiv.org/abs/1708.03731). The sklearn-bot will
execute a set of `n_executions`, each time selecting a task at random from the
full set of tasks. 

### Benchmark suite
To run the sklearn-bot on a a benchmark suite from OpenML, please use the
following command:

```
python examples/run_on_study.py --study_id OpenML100 --openml_server https://www.openml.org/api/v1/ --openml_apikey abcdef --classifier_name decision_tree --upload_result
```

This function has the same command line options as `run_on_task`, except for the
option `task_id`. Additionally,
* `study_id` (int or string) refers to the study ID on which the sklearn-bot
shall be ran. 

### Obtaining results
Usually, running the sklearn-bot is done so that the results can be re-used
in one or another way. Once the results have been stored on OpenML, it is 
important to be able to acquire them back. Although it is slightly out of the
scope of the sklearn-bot, the following command obtains all results that have 
been created using the given search space. Note that this can also include
results from other people that ran the sklearn-bot, or happened to run a
scikit-learn classifier with hyperparameter settings that also fell within the
search range. 

To obtain results from the sklearn-bot that were uploaded to OpenML (using the
`--upload_result` flag), please use the following command:

```
python examples/obtain_results.py --study_id OpenML100 --openml_server https://www.openml.org/api/v1/ --scoring predictive_accuracy --classifier_name decision_tree
```

The following command line options are accepted: 
* `output_directory`: This is where the results will be placed as ARFF file. 
Also cache files will be stored here, that allow for fast regeneration of the
datasets.
* `num_runs`: The number of runs per task that will be obtained. Setting this to
a number lower than the actual available runs will allow for efficient caching.
* `study_id`: Refers to the benchmark suite (which tasks will be included)
* `scoring`: The performance measure to download. Defaults to 
`predictive_accuracy`, but for example `area_under_roc_curve`, `f_measure` and 
`precision` are also sensible options. 
* `normalize` (flag): If set, all performance results will be normalized to the
interval [0, 1] per task. 
* `openml_server`: The server from which the results should be downloaded. Make
sure this is the same as the server to which the results where uploaded.
* `classifier_name`: The classifier from which the results should be downloaded.
Make sure that this is the same as the classifier with which the bot was ran. 

Note that there is no need to provide an API key, as the OpenML server is only
used for read operations. 

## Feature Requests

The following features will gradually be added to the sklearn-bot (contributors
are welcome):
* More classifiers. We want to extend the capabilities of the sklearn-bot to all 
configuration spaces in auto-sklearn. 
* Non-static pipelines. Although the current definition of a pipeline is fixed, 
we aim to add the notion of non-static pipelines. This should be incorporated
in a modular way. 
* More sampling methods. Currently, the sklearn-bot sampled uniformly from a set
of tasks, however it would be great if it was able to sample according to the
number of runs per task present on OpenML. 


## Dependencies
* [OpenML-Python](https://pypi.org/project/openml/) - Base functionality for
connecting with OpenML. 
* [OpenML-Python-Contrib](https://github.com/openml/openml-python-contrib/) - 
Not on pypi yet. Used for convenience functions to obtain the results from 
OpenML. The bot itself does not rely on this.
* [Scikit-learn](https://pypi.org/project/scikit-learn/) - Version 0.20.0 and
up.
* [ConfigSpace](https://pypi.org/project/ConfigSpace/) - For defining search 
spaces. 

