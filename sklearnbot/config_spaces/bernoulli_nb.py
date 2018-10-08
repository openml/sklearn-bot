import ConfigSpace


def get_hyperparameter_search_space(seed):
    """
    The Bernoulli NB configuration space based on the search space from
    auto-sklearn:
    https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/classification/bernoulli_nb.py

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('sklearn.naive_bayes.BernoulliNB', seed)

    imputation = ConfigSpace.hyperparameters.CategoricalHyperparameter(
        name='columntransformer__numeric__imputer__strategy', choices=['mean', 'median', 'most_frequent'])
    # the smoothing parameter is a non-negative float
    # I will limit it to 1000 and put it on a logarithmic scale. (SF)
    # Please adjust that, if you know a proper range, this is just a guess.
    alpha = ConfigSpace.UniformFloatHyperparameter(
        name='alpha', lower=1e-2, upper=100, default_value=1, log=True)
    fit_prior = ConfigSpace.CategoricalHyperparameter(
        name='fit_prior', choices=[True, False], default_value=True)

    cs.add_hyperparameters([
        imputation,
        alpha,
        fit_prior
    ])

    return cs
