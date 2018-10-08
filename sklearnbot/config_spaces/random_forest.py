import ConfigSpace


def get_hyperparameter_search_space(seed):
    """
    The random forest configuration space based on the search space from
    auto-sklearn:
    https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/classification/random_forest.py

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('sklearn.ensemble.RandomForestClassifier', seed)
    imputation = ConfigSpace.CategoricalHyperparameter(
        name='columntransformer__numeric__imputer__strategy', choices=['mean', 'median', 'most_frequent'])
    # TODO: parameterize the number of estimators?
    n_estimators = ConfigSpace.Constant(
        name='randomforestclassifier__n_estimators', value=100)
    criterion = ConfigSpace.CategoricalHyperparameter(
        name='randomforestclassifier__criterion', choices=['gini', 'entropy'], default_value='gini')
    max_features = ConfigSpace.UniformFloatHyperparameter(
        name='randomforestclassifier__max_features', lower=0., upper=1., default_value=0.5)
    # max_depth = ConfigSpace.UnParametrizedHyperparameter(
    #   name='randomforestclassifier__max_depth', value=None)
    min_samples_split = ConfigSpace.UniformIntegerHyperparameter(
        name='randomforestclassifier__min_samples_split', lower=2, upper=20, default_value=2)
    min_samples_leaf = ConfigSpace.UniformIntegerHyperparameter(
        name='randomforestclassifier__min_samples_leaf', lower=1, upper=20, default_value=1)
    min_weight_fraction_leaf = ConfigSpace.UnParametrizedHyperparameter(
        name='randomforestclassifier__min_weight_fraction_leaf', value=0.)
    # max_leaf_nodes = ConfigSpace.UnParametrizedHyperparameter(
    #   name='randomforestclassifier__max_leaf_nodes', value=None)
    bootstrap = ConfigSpace.CategoricalHyperparameter(
        name='randomforestclassifier__bootstrap', choices=[True, False], default_value=True)
    cs.add_hyperparameters([
        imputation,
        n_estimators,
        criterion,
        max_features,
        # max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        # max_leaf_nodes,
        bootstrap
    ])

    return cs
