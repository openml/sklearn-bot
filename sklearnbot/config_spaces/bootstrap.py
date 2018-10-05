import ConfigSpace
import sklearn
import sklearnbot
import typing


def get_random_configuration(configuration_space: ConfigSpace.ConfigurationSpace) \
        -> typing.Dict[str, typing.Union[int, float, str, None]]:
    """
    Samples a random configuration from the config space and corrects the names
    to scikit-learn hyperparameters.

    Parameters
    ----------
    configuration_space: ConfigSpace.ConfigurationSpace
        The configuration space to sample from

    Returns
    -------
    param_grid: dict[str, mixed]
        A configuration, represented as a dict mapping from the hyperparameter
        name to the hyperparameter value
    """
    original = configuration_space.sample_configuration(1).get_dictionary()
    param_grid = dict()
    for param_name, value in original.items():
        hyperparameter = configuration_space.get_hyperparameter(param_name)
        full_name = hyperparameter.meta['component'] + '__' + param_name
        param_grid[full_name] = value
    return param_grid


def get_available_config_spaces():
    """
    Returns a list of all available configuration spaces. To be used in
    example scripts, to determine which classifiers this can be ran with. 

    Returns
    -------
    config_spaces : list[str]
        A list of all available configuration spaces.
    """
    return ['decision_tree']


def get_config_space(classifier: sklearn.base.BaseEstimator, seed: typing.Optional[int]) \
        -> ConfigSpace.ConfigurationSpace:
    """
    Maps string names to a stored instantiation of the configuration space.

    Parameters
    ----------
    classifier: str
        The string name of the config space

    seed: int or None
        Will be passed to the Configuration Space object, and used for random
        sampling. Leave to None to assign a random seed (often preferred)

    Returns
    -------
    ConfigSpace.ConfigurationSpace
        An instantiation of the ConfigurationSpace
    """
    if classifier == 'decision_tree':
        return sklearnbot.config_spaces.decision_tree.get_hyperparameter_search_space(seed)
    else:
        raise ValueError()
