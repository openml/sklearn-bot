import ConfigSpace
import importlib
import sklearn
import sklearn.compose
import sklearn.impute
import typing


def deserialize(configuration_space: ConfigSpace.ConfigurationSpace,
                numeric_indices: typing.List[int],
                nominal_indices: typing.List[int])\
        -> sklearn.base.BaseEstimator:
    """
    Takes a ConfigSpace object and deserializes it back to an appropriate
    scikit-learn Pipeline. It will be a pipeline, containing a missing value
    indicator (numeric attributes only), imputer, scaler (numeric attributes
    only), onehotencoder (nominal attributes only) and the classifier.

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
    numeric_transformer = sklearn.pipeline.make_pipeline(
        sklearn.impute.MissingIndicator(error_on_new=False),
        sklearn.preprocessing.Imputer(),
        sklearn.preprocessing.StandardScaler())

    # note that the dataset is encoded numerically, hence we can only impute
    # numeric values, even for the categorical columns. 
    categorical_transformer = sklearn.pipeline.make_pipeline(
        sklearn.impute.SimpleImputer(strategy='constant', fill_value=-1),
        sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'))

    transformer = sklearn.compose.ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_indices),
            ('nominal', categorical_transformer, nominal_indices)],
        remainder='passthrough')

    # TODO: this should come from solid meta-data, rather than the name
    module_name = configuration_space.name.rsplit('.', 1)
    model_class = getattr(importlib.import_module(module_name[0]),
                          module_name[1])
    clf = sklearn.pipeline.make_pipeline(transformer, model_class())
    if configuration_space.meta is not None:
        clf.set_params(**configuration_space.meta)
    return clf
