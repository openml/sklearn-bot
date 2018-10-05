import ConfigSpace


def get_hyperparameter_search_space(seed):
    cs = ConfigSpace.ConfigurationSpace("sklearn.tree.DecisionTreeClassifier", seed)
    strategy = ConfigSpace.CategoricalHyperparameter(
        'strategy', ['mean', 'median'], default_value='median', meta={'component': 'columntransformer__numeric__imputer'})
    criterion = ConfigSpace.CategoricalHyperparameter(
        'criterion', ['gini', 'entropy'], default_value='gini', meta={'component': 'decisiontreeclassifier'})
    max_depth = ConfigSpace.UniformFloatHyperparameter(
        'max_depth', 0., 2., default_value=0.5, meta={'component': 'decisiontreeclassifier'})
    min_samples_split = ConfigSpace.UniformIntegerHyperparameter(
        'min_samples_split', 2, 20, default_value=2, meta={'component': 'decisiontreeclassifier'})
    min_samples_leaf = ConfigSpace.UniformIntegerHyperparameter(
        'min_samples_leaf', 1, 20, default_value=1, meta={'component': 'decisiontreeclassifier'})
    min_weight_fraction_leaf = ConfigSpace.Constant(
        'min_weight_fraction_leaf', 0.0, meta={'component': 'decisiontreeclassifier'})
    max_features = ConfigSpace.UnParametrizedHyperparameter(
        'max_features', 1.0, meta={'component': 'decisiontreeclassifier'})
    # max_leaf_nodes = ConfigSpace.UnParametrizedHyperparameter(
    #     'max_leaf_nodes', None, meta={'component': 'decisiontreeclassifier'})
    min_impurity_decrease = ConfigSpace.UnParametrizedHyperparameter(
        'min_impurity_decrease', 0.0, meta={'component': 'decisiontreeclassifier'})
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
