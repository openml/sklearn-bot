import ConfigSpace

from sklearnbot.config_spaces import ConfigSpaceWrapper


def get_hyperparameter_search_space(seed) -> ConfigSpaceWrapper:
    """
    k-NN search space based on a best effort using the scikit-learn
    implementation.

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('sklearn.neighbors.KNeighborsClassifier', seed)

    n_neighbors = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='n_neighbors', lower=1, upper=20, default_value=5)
    weights = ConfigSpace.hyperparameters.CategoricalHyperparameter(
        name='weights', choices=['uniform', 'distance'], default_value='uniform')
    algorithm = ConfigSpace.hyperparameters.CategoricalHyperparameter(
        name='algorithm', choices=['auto', 'ball_tree', 'kd_tree', 'brute'], default_value='auto')
    leaf_size = ConfigSpace.UniformIntegerHyperparameter(
        name='leaf_size', lower=1, upper=50, default_value=1)
    p = ConfigSpace.UniformIntegerHyperparameter(
        name='p', lower=1, upper=5, default_value=2)
    metric = ConfigSpace.hyperparameters.CategoricalHyperparameter(
        name='metric', choices=['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis'], default_value='minkowski')

    hyperparameters = [
        n_neighbors,
        weights,
        algorithm,
        leaf_size,
        p,
        metric,
    ]

    leaf_size_condition = ConfigSpace.InCondition(leaf_size, algorithm, ['ball_tree', 'kd_tree'])
    conditions = [leaf_size_condition]

    return ConfigSpaceWrapper(cs, hyperparameters, conditions)
