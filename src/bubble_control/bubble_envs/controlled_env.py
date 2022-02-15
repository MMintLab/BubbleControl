import numpy as np
import copy


class ControlledEnvWrapper(object):
    """
    Adds a controller on top of an environment so the action returned by get_action will be controlled.
    """

    def __init__(self, env, controller, random_action_prob=0, controlled_action_keys=None):
        """

        Args:
            env:
            controller:
            random_action_prob: <float> between [0,1], probability of sampling a random action
            controlled_action_keys: <list of string> representing the actions to control. The rest would be random. If None, all actions will be controlled.
        """
        self.env = env
        self.controller = controller
        self.random_action_prob = random_action_prob
        self.controlled_action_keys = controlled_action_keys
        self.observation = self.env.get_observation()

    @classmethod
    def get_name(cls):
        return 'controlled_env'

    def get_action(self):
        # We extend the original action dictionary to record weather or not the action came from a random or controlled policy
        random_action_p = np.random.random()
        random_action, valid_random_action = self.env.get_action()
        if random_action_p < self.random_action_prob:
            # Random action (sample env action space)
            random_action['action_controller'] = 'random'
            return random_action, valid_random_action
        else:
            # Use the controller
            self.observation = self.env.get_observation()
            controlled_action= self.controller.control(self.observation)
            # NOTE: Some controllers like MPPI have internal parameters that store previously computed information to improve sample efficiency.
            #       Here, we will not update any information when we have random actions.
            # pack the action with the right order
            valid_controlled_action = self.env.is_action_valid(controlled_action)
            controlled_action['action_controller'] = '{}'.format(self.controller.name)
            if self.controlled_action_keys is not None:
                random_action_keys = [k for k in random_action.keys() if k not in self.controlled_action_keys]
                for random_key in random_action_keys:
                    controlled_action[random_key] = random_action[random_key] # replace the controlled

            return controlled_action, valid_controlled_action

    # Wrap all other methods so we use the ones from self.env
    def __getattr__(self, attr):
        self_attr_name = '_{}_'.format(self.__class__.__name__)
        # print('Getting the attribute for: {}'.format(self_attr_name))
        if self_attr_name in attr:
            attr_name = attr.split(self_attr_name)[1]
            return getattr(self, attr_name)
        elif attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.env, attr)
