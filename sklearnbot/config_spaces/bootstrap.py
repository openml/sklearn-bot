import sklearnbot


def get_config_space(classifier, seed):
    if classifier == 'decision_tree':
        return sklearnbot.config_spaces.decision_tree.get_hyperparameter_search_space(seed)
    else:
        raise ValueError()
