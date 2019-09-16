import ConfigSpace

from sklearnbot.config_spaces import ConfigSpaceWrapper


def get_hyperparameter_search_space(seed) -> ConfigSpaceWrapper:
    """
    The SVM configuration space based on the search space from
    auto-sklearn:
    https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/classification/libsvm_svc.py

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('sklearn.svm.SVC', seed)

    C = ConfigSpace.UniformFloatHyperparameter(
        name='C', lower=0.03125, upper=32768, log=True, default_value=1.0)
    kernel = ConfigSpace.CategoricalHyperparameter(
        name='kernel', choices=['rbf', 'poly', 'sigmoid'], default_value='rbf')
    degree = ConfigSpace.UniformIntegerHyperparameter(
        name='degree', lower=1, upper=5, default_value=3)
    gamma = ConfigSpace.UniformFloatHyperparameter(
        name='gamma', lower=3.0517578125e-05, upper=8, log=True, default_value=0.1)
    coef0 = ConfigSpace.UniformFloatHyperparameter(
        name='coef0', lower=-1, upper=1, default_value=0)
    shrinking = ConfigSpace.CategoricalHyperparameter(
        name='shrinking', choices=[True, False], default_value=True)
    tol = ConfigSpace.UniformFloatHyperparameter(
        name='tol', lower=1e-5, upper=1e-1, default_value=1e-3, log=True)
    max_iter = ConfigSpace.UnParametrizedHyperparameter('svc__max_iter', -1)

    hyperparameters = [
        C,
        kernel,
        degree,
        gamma,
        coef0,
        shrinking,
        tol,
        max_iter
    ]

    degree_depends_on_poly = ConfigSpace.EqualsCondition(degree, kernel, 'poly')
    coef0_condition = ConfigSpace.InCondition(coef0, kernel, ['poly', 'sigmoid'])
    conditions = [degree_depends_on_poly, coef0_condition]

    return ConfigSpaceWrapper(cs, hyperparameters, conditions)
