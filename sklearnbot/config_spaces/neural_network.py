import ConfigSpace

from sklearnbot.config_spaces import ConfigSpaceWrapper


def get_hyperparameter_search_space(seed) -> ConfigSpaceWrapper:
    """
    Neural Network search space based on a best effort using the scikit-learn
    implementation. Note that for state of the art performance, other packages
    could be preferred.

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('sklearn.neural_network.MLPClassifier', seed)
    strategy = ConfigSpace.CategoricalHyperparameter(
        name='columntransformer__numeric__imputer__strategy', choices=['mean', 'median', 'most_frequent'])
    hidden_layer_sizes = ConfigSpace.UniformIntegerHyperparameter(
        name='mlpclassifier__hidden_layer_sizes', lower=32, upper=2048, default_value=2048)
    activation = ConfigSpace.CategoricalHyperparameter(
        name='mlpclassifier__activation', choices=['identity', 'logistic', 'tanh', 'relu'], default_value='relu')
    solver = ConfigSpace.CategoricalHyperparameter(
        name='mlpclassifier__solver', choices=['lbfgs', 'sgd', 'adam'], default_value='adam')
    alpha = ConfigSpace.UniformFloatHyperparameter(
        name='mlpclassifier__alpha', lower=1e-5, upper=1e-1, log=True, default_value=1e-4)
    batch_size = ConfigSpace.UniformIntegerHyperparameter(
        name='mlpclassifier__batch_size', lower=32, upper=4096, default_value=200)
    learning_rate = ConfigSpace.CategoricalHyperparameter(
        name='mlpclassifier__learning_rate', choices=['constant', 'invscaling', 'adaptive'], default_value='constant')
    learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
        name='mlpclassifier__learning_rate_init', lower=1e-5, upper=1e-1, log=True, default_value=1e-04)
    # TODO: Sensible range??
    power_t = ConfigSpace.UniformFloatHyperparameter(
        name='mlpclassifier__power_t', lower=1e-5, upper=1, log=True, default_value=0.5)
    max_iter = ConfigSpace.UniformIntegerHyperparameter(
        name='mlpclassifier__max_iter', lower=64, upper=1024, default_value=200)
    shuffle = ConfigSpace.CategoricalHyperparameter(
        name='mlpclassifier__shuffle', choices=[True, False], default_value=True)
    tol = ConfigSpace.UniformFloatHyperparameter(
        name='mlpclassifier__tol', lower=1e-5, upper=1e-1, default_value=1e-4, log=True)
    # TODO: log-scale?
    momentum = ConfigSpace.UniformFloatHyperparameter(
        name='mlpclassifier__momentum', lower=0, upper=1, default_value=0.9)
    nesterovs_momentum = ConfigSpace.CategoricalHyperparameter(
        name='mlpclassifier__nesterovs_momentum', choices=[True, False], default_value=True)
    early_stopping = ConfigSpace.CategoricalHyperparameter(
        name='mlpclassifier__early_stopping', choices=[True, False], default_value=True)
    validation_fraction = ConfigSpace.UniformFloatHyperparameter(
        name='mlpclassifier__validation_fraction', lower=0, upper=1, default_value=0.1)
    beta_1 = ConfigSpace.UniformFloatHyperparameter(
        name='mlpclassifier__beta_1', lower=0, upper=1, default_value=0.9)
    beta_2 = ConfigSpace.UniformFloatHyperparameter(
        name='mlpclassifier__beta_2', lower=0, upper=1, default_value=0.999)
    n_iter_no_change = ConfigSpace.UniformIntegerHyperparameter(
        name='mlpclassifier__n_iter_no_change', lower=1, upper=1024, default_value=200)

    hyperparameters = [
        strategy,
        hidden_layer_sizes,
        activation,
        solver,
        alpha,
        batch_size,
        learning_rate,
        learning_rate_init,
        power_t,
        max_iter,
        shuffle,
        tol,
        momentum,
        nesterovs_momentum,
        early_stopping,
        validation_fraction,
        beta_1,
        beta_2,
        n_iter_no_change,
    ]

    batch_size_condition = ConfigSpace.InCondition(batch_size, solver, ['sgd', 'adam'])
    learning_rate_init_condition = ConfigSpace.InCondition(learning_rate_init, solver, ['sgd', 'adam'])
    power_t_condition = ConfigSpace.EqualsCondition(power_t, solver, 'sgd')
    shuffle_confition = ConfigSpace.InCondition(shuffle, solver, ['sgd', 'adam'])
    tol_condition = ConfigSpace.InCondition(tol, learning_rate, ['constant', 'invscaling'])
    momentum_confition = ConfigSpace.EqualsCondition(momentum, solver, 'sgd')
    nesterovs_momentum_confition_solver = ConfigSpace.EqualsCondition(nesterovs_momentum, solver, 'sgd')
    nesterovs_momentum_confition_momentum = ConfigSpace.GreaterThanCondition(nesterovs_momentum, momentum, 0)
    nesterovs_momentum_conjunstion = ConfigSpace.AndConjunction(nesterovs_momentum_confition_solver,
                                                                nesterovs_momentum_confition_momentum)
    early_stopping_condition = ConfigSpace.InCondition(early_stopping, solver, ['sgd', 'adam'])
    validation_fraction_condition = ConfigSpace.EqualsCondition(validation_fraction, early_stopping, True)
    beta_1_condition = ConfigSpace.EqualsCondition(beta_1, solver, 'adam')
    beta_2_condition = ConfigSpace.EqualsCondition(beta_2, solver, 'adam')
    n_iter_no_change_condition_solver = ConfigSpace.InCondition(n_iter_no_change, solver, ['sgd', 'adam'])

    conditions = [
        batch_size_condition, learning_rate_init_condition, power_t_condition, shuffle_confition, tol_condition,
        momentum_confition, nesterovs_momentum_conjunstion, early_stopping_condition, validation_fraction_condition,
        beta_1_condition, beta_2_condition, n_iter_no_change_condition_solver
    ]

    return ConfigSpaceWrapper(cs, hyperparameters, conditions)
