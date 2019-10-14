import ConfigSpace
import copy
import typing


class ConfigSpaceWrapper(object):

    def __init__(self, config_space: ConfigSpace.ConfigurationSpace, hyperparameters: typing.List,
                 conditions: typing.Optional[typing.List]):
        self.config_space = config_space
        self.hyperparameters = hyperparameters
        self.conditions = conditions
        self.wrapped_in_pipeline = False

    def exclude_hyperparameter(self, name):
        hp_index = None
        for idx, hp in enumerate(self.hyperparameters):
            if hp.name == name:
                hp_index = idx
        del self.hyperparameters[hp_index]

    def reset_conditions(self):
        self.conditions = None

    def assemble(self) -> ConfigSpace.ConfigurationSpace:
        config_space = copy.deepcopy(self.config_space)
        config_space.add_hyperparameters(self.hyperparameters)
        if self.conditions is not None:
            config_space.add_conditions(self.conditions)
        return config_space

    def wrap_in_fixed_pipeline(self):
        if self.wrapped_in_pipeline:
            raise ValueError('Can not doubly wrap the fixed pipeline.')

        clf_name = self.config_space.name.rsplit('.', 1)[-1]
        for hyperparameter in self.hyperparameters:
            hyperparameter.name = '%s__%s' % (clf_name.lower(), hyperparameter.name)

        imputation = ConfigSpace.hyperparameters.CategoricalHyperparameter(
            name='columntransformer__numeric__imputer__strategy', choices=['mean', 'median', 'most_frequent'])
        self.config_space.add_hyperparameter(imputation)

        self.wrapped_in_pipeline = True
