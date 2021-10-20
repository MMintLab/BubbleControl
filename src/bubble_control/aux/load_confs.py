import yaml
import sys
import os
import bubble_utils

package_path = project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/bubble_control')[0], 'bubble_control')


def load_bubble_reconstruction_params():
    bubble_reconstruction_configs_path = os.path.join(package_path, 'config', 'bubble_reconstruction_params.yaml')
    bubble_reconstruction_configs = None
    with open(bubble_reconstruction_configs_path) as f:
        bubble_reconstruction_configs = yaml.load(f, Loader=yaml.SafeLoader)
    return bubble_reconstruction_configs
