import ConfigSpace
from sklearn.tree import DecisionTreeClassifier


def get_hyperparameter_search_space(seed):
    cs = ConfigSpace.ConfigurationSpace('sklearn.ensemble.AdaBoostClassifier',
                                        seed,
                                        meta={"adaboostclassifier__base_estimator": DecisionTreeClassifier()})

    imputation = ConfigSpace.hyperparameters.CategoricalHyperparameter(
        name='columntransformer__numeric__imputer__strategy', choices=['mean', 'median', 'most_frequent'])
    n_estimators = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name="adaboostclassifier__n_estimators", lower=50, upper=500, default_value=50, log=False)
    learning_rate = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name="adaboostclassifier__learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
    algorithm = ConfigSpace.hyperparameters.CategoricalHyperparameter(
        name="adaboostclassifier__algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")
    max_depth = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name="adaboostclassifier__base_estimator__max_depth", lower=1, upper=10, default_value=1, log=False)

    cs.add_hyperparameters([imputation, n_estimators, learning_rate, algorithm, max_depth])

    return cs
