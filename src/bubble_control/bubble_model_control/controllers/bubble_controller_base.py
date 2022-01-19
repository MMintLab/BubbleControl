import abc


class BubbleControllerBase(abc.ABC):

    def control(self, state_sample):
        action = self._query_controller(state_sample)
        return action

    @abc.abstractmethod
    def _query_controller(self, state_sample):
        pass


class BubbleModelController(BubbleControllerBase):

    def __init__(self, model, env, object_pose_estimator, cost_function):
        self.model = model
        self.env = env
        self.object_pose_estimator = object_pose_estimator
        self.cost_function = cost_function



