import ConfigSpace
import importlib
import sklearn
import sklearn.compose
import sklearn.impute
import sklearnbot
import typing


def deserialize(configuration_space: ConfigSpace.ConfigurationSpace,
                numeric_indices: typing.List,
                categorical_indices: typing.List):
    numeric_transformer = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.Imputer(),
        sklearn.preprocessing.StandardScaler())

    categorical_transformer = sklearn.pipeline.make_pipeline(
        sklearn.impute.SimpleImputer(strategy='constant', fill_value='missing'),
        sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'))

    transformer = sklearn.compose.ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_indices),
            ('nominal', categorical_transformer, categorical_indices)],
        remainder='passthrough')

    # TODO: this should come from solid meta-data, rather than the name
    module_name = configuration_space.name.rsplit('.', 1)
    model_class = getattr(importlib.import_module(module_name[0]),
                          module_name[1])
    configuration_dict = sklearnbot.config_spaces.get_random_configuration(configuration_space)
    clf = sklearn.pipeline.make_pipeline(transformer, model_class())
    print(configuration_dict)
    clf.set_params(**configuration_dict)
    return clf
