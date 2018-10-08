import ConfigSpace
import random
import sklearnbot
import typing


ALL_WILDCARD_NAME = 'all'


def get_available_config_spaces(allow_all: bool):
    """
    Returns a list of all available configuration spaces. To be used in
    example scripts, to determine which classifiers this can be ran with.

    Parameters
    ----------
    allow_all: bool
        If set to true, the wildcard value `all` will be added to this list

    Returns
    -------
    config_spaces : list[str]
        A list of all available configuration spaces.
    """
    config_spaces = [
        'adaboost',
        'bernoulli_nb',
        'decision_tree',
        'extra_trees',
        'gradient_boosting',
        'knn',
        'neural_network',
        'random_forest',
        'sgd',
        'svc'
    ]
    if allow_all:
        config_spaces = [ALL_WILDCARD_NAME] + config_spaces
    return config_spaces


def get_config_space(classifier_name: str, seed: typing.Optional[int]) \
        -> ConfigSpace.ConfigurationSpace:
    """
    Maps string names to a stored instantiation of the configuration space.

    Parameters
    ----------
    classifier_name: str
        The string name of the config space

    seed: int or None
        Will be passed to the Configuration Space object, and used for random
        sampling. Leave to None to assign a random seed (often preferred)

    Returns
    -------
    ConfigSpace.ConfigurationSpace
        An instantiation of the ConfigurationSpace
    """
    if classifier_name == ALL_WILDCARD_NAME:
        classifier_name = random.choice(get_available_config_spaces(False))
    if classifier_name not in get_available_config_spaces(False):
        raise ValueError('Classifier search space not implemented: %s' % classifier_name)
    return getattr(sklearnbot.config_spaces, classifier_name).get_hyperparameter_search_space(seed)
