import ConfigSpace
import importlib
import scipy.stats
import sklearn
import sklearn.compose
import sklearn.feature_selection
import sklearn.impute
import sklearn.model_selection
import typing


def _config_space_to_parameter_distributions(
        configuration_space: ConfigSpace.ConfigurationSpace) \
        -> typing.Dict[str, typing.Union[typing.List,
                                         scipy.stats.rv_discrete,
                                         scipy.stats.rv_continuous]]:
    """
    Takes a ConfigSpace object and serializes it into a parameter grid, to be
    used by the scikit-learn interface.

    Parameters
    ----------
    configuration_space: ConfigSpace.ConfigurationSpace
        The configuration space describes all hyperparameters and ranges

    Returns
    -------
    result: Dict
        A dict mapping from hyperparameter name to List of Values or distribution
    """
    result = dict()
    for hyperparameter in configuration_space.get_hyperparameters():
        if isinstance(hyperparameter, ConfigSpace.hyperparameters.UniformFloatHyperparameter):
            loc = hyperparameter.lower
            scale = hyperparameter.upper - hyperparameter.lower
            distribution = scipy.stats.uniform(loc=loc, scale=scale)
        elif isinstance(hyperparameter, ConfigSpace.hyperparameters.UniformIntegerHyperparameter):
            distribution = scipy.stats.randint(hyperparameter.lower, hyperparameter.upper + 1)
        elif isinstance(hyperparameter, ConfigSpace.hyperparameters.CategoricalHyperparameter):
            distribution = list(hyperparameter.choices)
        elif isinstance(hyperparameter, ConfigSpace.hyperparameters.UnParametrizedHyperparameter):
            distribution = [hyperparameter.value]
        elif isinstance(hyperparameter, ConfigSpace.hyperparameters.Constant):
            distribution = [hyperparameter.value]
        else:
            raise ValueError('Hyperparameter type not supported yet: %s' % type(hyperparameter))
        result[hyperparameter.name] = distribution
    return result


def _seed_models(model):
    rs_params = dict()
    for param, value in model.get_params(deep=True).items():
        if param.endswith('random_state') and value is None:
            rs_params[param] = 0
    model.set_params(**rs_params)
    return model


def as_estimator(configuration_space: ConfigSpace.ConfigurationSpace, skip_meta: bool):
    """
    Takes a ConfigSpace object and deserializes it back to an appropriate
    scikit-learn classifier.

    Parameters
    ----------
    configuration_space: ConfigSpace.ConfigurationSpace
        The configuration space that holds the information to instantiate the
        classifier

    skip_meta: bool
        If set to true, additional meta-features as defined by the config space will
        not be set

    Returns
    -------
    clf: sklearn.BaseEstimator
        The instantiated classifier with default hyperparameters
    """
    # TODO: this should come from solid meta-data, rather than the name
    module_name = configuration_space.name.rsplit('.', 1)
    clf = getattr(importlib.import_module(module_name[0]), module_name[1])()
    if not skip_meta and configuration_space.meta is not None:
        clf.set_params(**configuration_space.meta)
    clf = _seed_models(clf)
    return clf


def as_pipeline(configuration_space: ConfigSpace.ConfigurationSpace,
                numeric_indices: typing.List[int],
                nominal_indices: typing.List[int])\
        -> sklearn.base.BaseEstimator:
    """
    Takes a ConfigSpace object and deserializes it back to an appropriate
    scikit-learn Pipeline. It will be a pipeline, containing an
    imputer, scaler (numeric attributes only), onehotencoder (nominal
    attributes only), variance threshold and the classifier.

    Parameters
    ----------
    configuration_space: ConfigSpace.ConfigurationSpace
        The configuration space that holds the information to instantiate the
        classifier

    numeric_indices: list[int]
        A numeric list indicating which attribute indices are numeric

    nominal_indices: list[int]
        A numeric list indicating which attribute indices are nominal

    Returns
    -------
    clf: sklearn.BaseEstimator
        The instantiated classifier with default hyperparameters
    """

    transformer = sklearn.compose.ColumnTransformer(
        transformers=[
            ('numeric', sklearn.preprocessing.StandardScaler(), numeric_indices),
            ('nominal', sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'), nominal_indices)],
        remainder='passthrough')

    clf = as_estimator(configuration_space, True)  # supposed to set random_state
    pipeline = sklearn.pipeline.make_pipeline(
        sklearn.impute.SimpleImputer(strategy='constant', fill_value=-1),
        transformer,
        sklearn.feature_selection.VarianceThreshold(),
        clf)
    if configuration_space.meta is not None:
        pipeline.set_params(**configuration_space.meta)
    return pipeline


def as_search_cv(configuration_space: ConfigSpace.ConfigurationSpace,
                 numeric_indices: typing.List[int],
                 nominal_indices: typing.List[int], **kwargs)\
        -> sklearn.model_selection.RandomizedSearchCV:
    """
    Takes a ConfigSpace object and deserializes it back to an appropriate
    scikit-learn Pipeline, wrapping it in a RandomizedSearchCV object.
    Note that this interface is experimental, and does not contain support for
    log-scale parameters.

    Parameters
    ----------
    configuration_space: ConfigSpace.ConfigurationSpace
        The configuration space that holds the information to instantiate the
        classifier

    numeric_indices: list[int]
        A numeric list indicating which attribute indices are numeric

    nominal_indices: list[int]
        A numeric list indicating which attribute indices are nominal

    Returns
    -------
    clf: sklearn.BaseEstimator
        The instantiated classifier with default hyperparameters
    """
    classifier = as_pipeline(configuration_space, numeric_indices, nominal_indices)
    param_dist = _config_space_to_parameter_distributions(configuration_space)
    search = sklearn.model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_dist,
        random_state=0,
        **kwargs
    )
    return search
