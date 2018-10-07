import ConfigSpace


def get_hyperparameter_search_space(seed: int) -> ConfigSpace.ConfigurationSpace:
    """
    The decision tree configuration space as presented in auto-sklearn.

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
    strategy = ConfigSpace.CategoricalHyperparameter(
        'columntransformer__numeric__imputer__strategy', ['mean', 'median'], default_value='median')
    criterion = ConfigSpace.CategoricalHyperparameter(
        'decisiontreeclassifier__criterion', ['gini', 'entropy'], default_value='gini')
    max_depth = ConfigSpace.UniformFloatHyperparameter(
        'decisiontreeclassifier__max_depth', 0., 2., default_value=0.5)
    min_samples_split = ConfigSpace.UniformIntegerHyperparameter(
        'decisiontreeclassifier__min_samples_split', 2, 20, default_value=2)
    min_samples_leaf = ConfigSpace.UniformIntegerHyperparameter(
        'decisiontreeclassifier__min_samples_leaf', 1, 20, default_value=1)
    min_weight_fraction_leaf = ConfigSpace.Constant(
        'decisiontreeclassifier__min_weight_fraction_leaf', 0.0)
    max_features = ConfigSpace.UnParametrizedHyperparameter(
        'decisiontreeclassifier__max_features', 1.0)
    # max_leaf_nodes = ConfigSpace.UnParametrizedHyperparameter(
    #     'max_leaf_nodes', None, meta={'component': 'decisiontreeclassifier'})
    min_impurity_decrease = ConfigSpace.UnParametrizedHyperparameter(
        'decisiontreeclassifier__min_impurity_decrease', 0.0)
    # TODO: max_leaf_nodes one can only be tuned once config space allows for this.

    cs.add_hyperparameters([strategy,
                            criterion,
                            max_features,
                            max_depth,
                            min_samples_split,
                            min_samples_leaf,
                            min_weight_fraction_leaf,
                            # max_leaf_nodes,
                            min_impurity_decrease,
                            ])
    return cs
