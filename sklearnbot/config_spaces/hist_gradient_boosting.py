import ConfigSpace


def get_hyperparameter_search_space(seed):
    """
    Histogram Gradient Boosting search space based on a best effort using the scikit-learn
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
    cs = ConfigSpace.ConfigurationSpace('sklearn.ensemble.HistGradientBoostingClassifier', seed)

    imputation = ConfigSpace.hyperparameters.CategoricalHyperparameter(
        name='columntransformer__numeric__imputer__strategy', choices=['mean', 'median', 'most_frequent'])

    loss = ConfigSpace.hyperparameters.Constant(name='histgradientboostingclassifier__loss', value='auto')

    learning_rate = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='histgradientboostingclassifier__learning_rate', lower=0.001, upper=1, default_value=1, log=True)

    max_iter = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='histgradientboostingclassifier__max_iter', lower=50, upper=500, default_value=100)

    max_leaf_nodes = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='histgradientboostingclassifier__max_leaf_nodes', lower=2, upper=256, default_value=31)  

    max_depth = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='histgradientboostingclassifier__max_depth', lower=2, upper=20, default_value=None)

    min_samples_leaf = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='histgradientboostingclassifier__min_samples_leaf', lower=1, upper=20, default_value=20)

    l2_regularization = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='histgradientboostingclassifier__l2_regularization', lower=1e-10, upper=1, default_value=0.0, log=True)
    
    max_bins = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='histgradientboostingclassifier__max_bins', lower=2, upper=512, default_value=256)

    validation_fraction = ConfigSpace.UniformFloatHyperparameter(
        name='histgradientboostingclassifier__validation_fraction', lower=0.1, upper=0.3, default_value=0.1)

    n_iter_no_change = ConfigSpace.UniformIntegerHyperparameter(
        name='histgradientboostingclassifier__n_iter_no_change', lower=1, upper=2048, default_value=None)

    tol = ConfigSpace.UniformFloatHyperparameter(
        name='histgradientboostingclassifier__tol', lower=1e-7, upper=1e-1, default_value=1e-7, log=True)

    cs.add_hyperparameters([
        imputation,
        loss,
        learning_rate,
        max_iter,
        max_leaf_nodes,
        max_depth,
        min_samples_leaf,
        l2_regularization,
        max_bins,
        min_weight_fraction_leaf,
        validation_fraction,
        n_iter_no_change,
        tol,
    ])

    return cs
