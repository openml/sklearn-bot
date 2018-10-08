# contents of this library should be moved to other library
import sklearnbot.config_spaces.adaboost
import sklearnbot.config_spaces.decision_tree
import sklearnbot.config_spaces.gradient_boosting
import sklearnbot.config_spaces.neural_network
import sklearnbot.config_spaces.random_forest
import sklearnbot.config_spaces.sgd
import sklearnbot.config_spaces.svc
from .bootstrap import get_available_config_spaces, get_config_space
