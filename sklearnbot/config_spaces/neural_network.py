import ConfigSpace


def get_hyperparameter_search_space(seed):
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
        name='columntransformer__numeric__imputer__strategy', choices=['mean', 'median'])
    hidden_layer_sizes = ConfigSpace.UniformIntegerHyperparameter(
        name='mlpclassifier__hidden_layer_sizes', lower=32, upper=2048, default_value=2048)
    activation = ConfigSpace.CategoricalHyperparameter(
        name='mlpclassifier__activation', choices=['identity', 'logistic', 'tanh', 'relu'], default='relu')
    solver = ConfigSpace.CategoricalHyperparameter(
        name='mlpclassifier__solver', choices=['lbfgs', 'sgd', 'adam'], default='adam')
    alpha = ConfigSpace.UniformFloatHyperparameter(
        name='mlpclassifier__alpha', lower=1e-05, upper=1e-01, log=True, default_value=1e-04)
    batch_size = ConfigSpace.UniformIntegerHyperparameter(
        name='mlpclassifier__batch_size', lower=32, upper=4096, default_value=200)
    learning_rate = ConfigSpace.CategoricalHyperparameter(
        name='mlpclassifier__learning_rate', choices=['constant', 'invscaling', 'adaptive'], default='constrant')
    learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
        name='mlpclassifier__learning_rate_init', lower=1e-05, upper=1e-01, log=True, default_value=1e-04)

    cs.add_hyperparameters([
        strategy
    ])

    batch_size_condition = ConfigSpace.InCondition(batch_size, solver, ['sgd', 'adam'])
    learning_rate_init_condition = ConfigSpace.InCondition(learning_rate_init, solver, ['sgd', 'adam'])

    return cs
