import ConfigSpace

from sklearnbot.config_spaces import ConfigSpaceWrapper
from sklearn.tree import DecisionTreeClassifier


def get_hyperparameter_search_space(seed) -> ConfigSpaceWrapper:
    """
    The adaboost configuration space based on the search space from
    auto-sklearn:
    https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/classification/adaboost.py

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('sklearn.ensemble.AdaBoostClassifier',
                                        seed,
                                        meta={"base_estimator": DecisionTreeClassifier(random_state=0)})

    n_estimators = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name="n_estimators", lower=50, upper=500, default_value=50, log=False)
    learning_rate = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
    algorithm = ConfigSpace.hyperparameters.CategoricalHyperparameter(
        name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")
    max_depth = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name="base_estimator__max_depth", lower=1, upper=10, default_value=1, log=False)

    hyperparameters = [n_estimators, learning_rate, algorithm, max_depth]

    return ConfigSpaceWrapper(cs, hyperparameters, None)
