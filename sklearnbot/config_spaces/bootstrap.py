import ConfigSpace
import sklearn
import sklearnbot
import typing


def get_random_configuration(configuration_space: ConfigSpace.ConfigurationSpace):
    original = configuration_space.sample_configuration(1).get_dictionary()
    transformed = dict()
    for param_name, value in original.items():
        hyperparameter = configuration_space.get_hyperparameter(param_name)
        full_name = hyperparameter.meta['component'] + '__' + param_name
        transformed[full_name] = value
    return transformed


def get_config_space(classifier: sklearn.base.BaseEstimator, seed: typing.Optional[int]):
    if classifier == 'decision_tree':
        return sklearnbot.config_spaces.decision_tree.get_hyperparameter_search_space(seed)
    else:
        raise ValueError()
