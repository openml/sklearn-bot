import ConfigSpace


def get_hyperparameter_search_space(seed):
    """
    The random forest configuration space based on the search space from
    auto-sklearn:
    https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/classification/sgd.py

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('sklearn.linear_model.SGDClassifier', seed)

    imputation = ConfigSpace.hyperparameters.CategoricalHyperparameter(
        name='columntransformer__numeric__imputer__strategy', choices=['mean', 'median', 'most_frequent'])
    loss = ConfigSpace.CategoricalHyperparameter(
        name='sgdclassifier__loss', choices=['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], default_value='log')
    penalty = ConfigSpace.CategoricalHyperparameter(
        name='sgdclassifier__penalty', choices=['l1', 'l2', 'elasticnet'], default_value='l2')
    alpha = ConfigSpace.UniformFloatHyperparameter(
        name='sgdclassifier__alpha', lower=1e-7, upper=1e-1, log=True, default_value=0.0001)
    l1_ratio = ConfigSpace.UniformFloatHyperparameter(
        name='sgdclassifier__l1_ratio', lower=1e-9, upper=1,  log=True, default_value=0.15)
    # fit_intercept = ConfigSpace.UnParametrizedHyperparameter(name='fit_intercept', value=True)
    tol = ConfigSpace.UniformFloatHyperparameter(
        name='sgdclassifier__tol', lower=1e-5, upper=1e-1, log=True, default_value=1e-4)
    epsilon = ConfigSpace.UniformFloatHyperparameter(
        name='sgdclassifier__epsilon', lower=1e-5, upper=1e-1, default_value=1e-4, log=True)
    learning_rate = ConfigSpace.CategoricalHyperparameter(
        name='sgdclassifier__learning_rate', choices=['optimal', 'invscaling', 'constant'], default_value='invscaling')
    eta0 = ConfigSpace.UniformFloatHyperparameter(
        name='sgdclassifier__eta0', lower=1e-7, upper=1e-1, default_value=0.01, log=True)
    power_t = ConfigSpace.UniformFloatHyperparameter('power_t', 1e-5, 1, default_value=0.5)
    average = ConfigSpace.CategoricalHyperparameter(
        name='sgdclassifier__average', choices=[False, True], default_value=False)

    cs.add_hyperparameters([
        imputation,
        loss,
        penalty,
        alpha,
        l1_ratio,
        # fit_intercept,
        tol,
        epsilon,
        learning_rate,
        eta0,
        power_t,
        average
    ])

    # TODO MF: add passive/aggressive here, although not properly documented?
    elasticnet = ConfigSpace.EqualsCondition(l1_ratio, penalty, 'elasticnet')
    epsilon_condition = ConfigSpace.EqualsCondition(epsilon, loss, 'modified_huber')

    power_t_condition = ConfigSpace.EqualsCondition(power_t, learning_rate, 'invscaling')

    # eta0 is only relevant if learning_rate!='optimal' according to code
    # https://github.com/scikit-learn/scikit-learn/blob/0.19.X/sklearn/
    # linear_model/sgd_fast.pyx#L603
    eta0_in_inv_con = ConfigSpace.InCondition(eta0, learning_rate, ['invscaling', 'constant'])

    cs.add_conditions([
        elasticnet,
        epsilon_condition,
        power_t_condition,
        eta0_in_inv_con
    ])

    return cs
