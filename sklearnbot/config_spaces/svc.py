import ConfigSpace


def get_hyperparameter_search_space(seed):
    cs = ConfigSpace.ConfigurationSpace('sklearn.svm.SVC', seed)

    imputation = ConfigSpace.CategoricalHyperparameter(
        'columntransformer__numeric__imputer__strategy', ['mean', 'median', 'most_frequent'])
    C = ConfigSpace.UniformFloatHyperparameter(
        'svc__C', 0.03125, 32768, log=True, default_value=1.0)
    kernel = ConfigSpace.CategoricalHyperparameter(
        name='svc__kernel', choices=['rbf', 'poly', 'sigmoid'], default_value='rbf')
    degree = ConfigSpace.UniformIntegerHyperparameter(
        'svc__degree', 1, 5, default_value=3)
    gamma = ConfigSpace.UniformFloatHyperparameter(
        'svc__gamma', 3.0517578125e-05, 8, log=True, default_value=0.1)
    coef0 = ConfigSpace.UniformFloatHyperparameter(
        'svc__coef0', -1, 1, default_value=0)
    shrinking = ConfigSpace.CategoricalHyperparameter(
        'svc__shrinking', [True, False], default_value=True)
    tol = ConfigSpace.UniformFloatHyperparameter(
        'svc__tol', 1e-5, 1e-1, default_value=1e-3, log=True)
    max_iter = ConfigSpace.UnParametrizedHyperparameter('svc__max_iter', -1)

    cs.add_hyperparameters([
        imputation,
        C,
        kernel,
        degree,
        gamma,
        coef0,
        shrinking,
        tol,
        max_iter
    ])

    degree_depends_on_poly = ConfigSpace.EqualsCondition(degree, kernel, 'poly')
    coef0_condition = ConfigSpace.InCondition(coef0, kernel, ['poly', 'sigmoid'])
    cs.add_condition(degree_depends_on_poly)
    cs.add_condition(coef0_condition)

    return cs
