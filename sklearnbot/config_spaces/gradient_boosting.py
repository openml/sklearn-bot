import ConfigSpace

from sklearnbot.config_spaces import ConfigSpaceWrapper


def get_hyperparameter_search_space(seed) -> ConfigSpaceWrapper:
    """
    Gradient Boosting search space based on a best effort using the scikit-learn
    implementation. Note that for state of the art performance, other packages,
    such as xgboost, could be preferred.

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('sklearn.ensemble.GradientBoostingClassifier', seed)

    imputation = ConfigSpace.hyperparameters.CategoricalHyperparameter(
        name='columntransformer__numeric__imputer__strategy', choices=['mean', 'median', 'most_frequent'])
    # fixed to deviance, as exponential requires two classes
    loss = ConfigSpace.hyperparameters.Constant(name='gradientboostingclassifier__loss', value='deviance')
    # JvR: changed after conversation with AM on 2019-01-17
    learning_rate = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='gradientboostingclassifier__learning_rate', lower=0.00001, upper=0.1, default_value=0.0001, log=True)
    n_estimators = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='gradientboostingclassifier__n_estimators', lower=64, upper=2048, default_value=100, log=True)
    subsample = ConfigSpace.UniformFloatHyperparameter(
        name='gradientboostingclassifier__subsample', lower=0.0, upper=1.0, default_value=1.0)
    criterion = ConfigSpace.hyperparameters.CategoricalHyperparameter(
        name='gradientboostingclassifier__criterion', choices=['friedman_mse', 'mse', 'mae'])
    min_samples_split = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='gradientboostingclassifier__min_samples_split', lower=2, upper=20, default_value=2)
    min_samples_leaf = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='gradientboostingclassifier__min_samples_leaf', lower=1, upper=20, default_value=1)
    # TODO: upper bound?
    min_weight_fraction_leaf = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='gradientboostingclassifier__min_weight_fraction_leaf', lower=0.0, upper=0.5, default_value=0.0)
    # JvR: changed after conversation with AM on 2019-01-17
    max_depth = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='gradientboostingclassifier__max_depth', lower=1, upper=32, default_value=3)
    # TODO: upper bound?
    min_impurity_decrease = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='gradientboostingclassifier__min_impurity_decrease', lower=0.0, upper=1.0, default_value=0.0)
    max_features = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='gradientboostingclassifier__max_features', lower=0.0, upper=1.0, default_value=0.0)
    validation_fraction = ConfigSpace.UniformFloatHyperparameter(
        name='gradientboostingclassifier__validation_fraction', lower=0, upper=1, default_value=0.1)
    n_iter_no_change = ConfigSpace.UniformIntegerHyperparameter(
        name='gradientboostingclassifier__n_iter_no_change', lower=1, upper=2048, default_value=200)
    tol = ConfigSpace.UniformFloatHyperparameter(
        name='gradientboostingclassifier__tol', lower=1e-5, upper=1e-1, default_value=1e-4, log=True)

    hyperparameters = [
        imputation,
        loss,
        learning_rate,
        n_estimators,
        subsample,
        criterion,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_depth,
        min_impurity_decrease,
        max_features,
        validation_fraction,
        n_iter_no_change,
        tol,
    ]

    return ConfigSpaceWrapper(cs, hyperparameters, None)
