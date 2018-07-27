import importlib
import sklearn
import sklearn.compose
import sklearn.impute


def deserialize(configuration_space, numeric_indices, categorical_indices):
    configuration = configuration_space.sample_configuration(1)

    numeric_transformer = sklearn.pipeline.make_pipeline(
        sklearn.impute.SimpleImputer(),
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

    clf = sklearn.pipeline.make_pipeline(transformer, model_class())
    clf.set_params(**configuration.get_dictionary())
    return clf
