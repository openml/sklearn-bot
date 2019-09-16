import ConfigSpace
import copy
import typing


class ConfigSpaceWrapper(object):

    def __init__(self, config_space: ConfigSpace.ConfigurationSpace, hyperparameters: typing.List,
                 conditions: typing.Optional[typing.List]):
        self.config_space = config_space
        self.hyperparameters = hyperparameters
        self.conditions = conditions

    def assemble(self) -> ConfigSpace.ConfigurationSpace:
        config_space = copy.deepcopy(self.config_space)
        config_space.add_hyperparameters(self.hyperparameters)
        if self.conditions is not None:
            config_space.add_conditions(self.conditions)
        return config_space
