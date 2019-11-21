import ConfigSpace

from sklearnbot.config_spaces import ConfigSpaceWrapper


def get_hyperparameter_search_space(seed: int) -> ConfigSpaceWrapper:
    """
    The extra trees configuration space based on the search space from
    auto-sklearn:
    https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/classification/extra_trees.py

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('sklearn.ensemble.ExtraTreesClassifier', seed)

    # TODO: parameterize the number of estimators?
    n_estimators = ConfigSpace.Constant(name='extratreesclassifier__n_estimators', value=100)
    criterion = ConfigSpace.CategoricalHyperparameter(
        name='criterion', choices=['gini', 'entropy'], default_value='gini')

    # The maximum number of features used in the forest is calculated as m^max_features, where
    # m is the total number of features, and max_features is the hyperparameter specified below.
    # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
    # corresponds with Geurts' heuristic.
    max_features = ConfigSpace.UniformFloatHyperparameter(
        name='max_features', lower=0., upper=1., default_value=0.5)
    # max_depth = ConfigSpace.UnParametrizedHyperparameter(name='extratreesclassifier__max_depth', value=None)
    min_samples_split = ConfigSpace.UniformIntegerHyperparameter(
        name='min_samples_split', lower=2, upper=20, default_value=2)
    min_samples_leaf = ConfigSpace.UniformIntegerHyperparameter(
        name='min_samples_leaf', lower=1, upper=20, default_value=1)
    min_weight_fraction_leaf = ConfigSpace.UnParametrizedHyperparameter(name='min_weight_fraction_leaf', value=0.)
    # max_leaf_nodes = ConfigSpace.UnParametrizedHyperparameter(name='max_leaf_nodes', value=None)
    min_impurity_decrease = ConfigSpace.UnParametrizedHyperparameter(name='min_impurity_decrease', value=0.0)

    bootstrap = ConfigSpace.CategoricalHyperparameter('bootstrap', [True, False], default_value=False)

    hyperparameters = [
        n_estimators,
        criterion,
        max_features,
        # max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        # max_leaf_nodes,
        min_impurity_decrease,
        bootstrap
    ]

    return ConfigSpaceWrapper(cs, hyperparameters, None)
