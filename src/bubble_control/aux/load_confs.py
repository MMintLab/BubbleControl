import yaml
import sys
import os
import bubble_utils

package_path = project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/bubble_control')[0], 'bubble_control')


def _load_config_from_path(path):
    config = None
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def load_bubble_reconstruction_params():
    bubble_reconstruction_configs_path = os.path.join(package_path, 'config', 'bubble_reconstruction_params.yaml')
    bubble_reconstruction_configs = _load_config_from_path(bubble_reconstruction_configs_path)
    return bubble_reconstruction_configs


def load_plane_params():
    bubble_reconstruction_configs_path = os.path.join(package_path, 'config', 'plane_params.yaml')
    bubble_reconstruction_configs = _load_config_from_path(bubble_reconstruction_configs_path)
    return bubble_reconstruction_configs

