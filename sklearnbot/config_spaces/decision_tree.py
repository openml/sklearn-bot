import ConfigSpace

from sklearnbot.config_spaces import ConfigSpaceWrapper


def get_hyperparameter_search_space(seed: int) -> ConfigSpaceWrapper:
    """
    The decision tree configuration space based on the search space from
    auto-sklearn:
    https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/classification/decision_tree.py

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace("sklearn.tree.DecisionTreeClassifier", seed)
    criterion = ConfigSpace.CategoricalHyperparameter(
        name='criterion', choices=['gini', 'entropy'], default_value='gini')
    min_samples_split = ConfigSpace.UniformIntegerHyperparameter(
        name='min_samples_split', lower=2, upper=20, default_value=2)
    min_samples_leaf = ConfigSpace.UniformIntegerHyperparameter(
        name='min_samples_leaf', lower=1, upper=20, default_value=1)
    min_weight_fraction_leaf = ConfigSpace.Constant(
        name='min_weight_fraction_leaf', value=0.0)
    max_features = ConfigSpace.UnParametrizedHyperparameter(
        name='max_features', value=1.0)
    min_impurity_decrease = ConfigSpace.UnParametrizedHyperparameter(
        'min_impurity_decrease', 0.0)
    # TODO: max_leaf_nodes one can only be tuned once config space allows for this.

    hyperparameters = [
        criterion, max_features, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, min_impurity_decrease
    ]
    return ConfigSpaceWrapper(cs, hyperparameters, None)
