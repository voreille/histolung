import yaml

from easydict import EasyDict as edict


def yaml_load(fileName):
    dict_config = None
    with open(fileName, 'r') as ymlfile:
        dict_config = edict(yaml.safe_load(ymlfile))

    return dict_config
