import ConfigSpace


def get_hyperparameter_search_space(seed: int) -> ConfigSpace.ConfigurationSpace:
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
    strategy = ConfigSpace.CategoricalHyperparameter(
        name='columntransformer__numeric__imputer__strategy', choices=['mean', 'median', 'most_frequent'], default_value='median')
    criterion = ConfigSpace.CategoricalHyperparameter(
        name='decisiontreeclassifier__criterion', choices=['gini', 'entropy'], default_value='gini')
    max_depth = ConfigSpace.UniformFloatHyperparameter(
        name='decisiontreeclassifier__max_depth', lower=0., upper=2., default_value=0.5)
    min_samples_split = ConfigSpace.UniformIntegerHyperparameter(
        name='decisiontreeclassifier__min_samples_split', lower=2, upper=20, default_value=2)
    min_samples_leaf = ConfigSpace.UniformIntegerHyperparameter(
        name='decisiontreeclassifier__min_samples_leaf', lower=1, upper=20, default_value=1)
    min_weight_fraction_leaf = ConfigSpace.Constant(
        name='decisiontreeclassifier__min_weight_fraction_leaf', value=0.0)
    max_features = ConfigSpace.UnParametrizedHyperparameter(
        name='decisiontreeclassifier__max_features', value=1.0)
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
