from gomoku_zero.config import Config


class GomokuModelAPI:
    def __init__(self, config: Config, agent_model):
        '''
        API for model prediction
        :param config:
        :param agent_model:
        '''
        self.config = config
        self.agent_model = agent_model

    def predict(self, x):
        '''
        outputs the policy and value functions
        :param x: the input batched states
        :return:
        '''
        assert x.ndim in (3, 4)
        #Changed
        #assert x.shape == (2, 6, 7) or x.shape[1:] == (2, 6, 7)
        assert x.shape == (2, 8, 5) or x.shape[1:] == (2, 8, 5)
        orig_x = x
        if x.ndim == 3:
            x = x.reshape(1, 2, 8, 5)
        policy, value = self.agent_model.model.predict_on_batch(x)

        if orig_x.ndim == 3:
            return policy[0], value[0]
        else:
            return policy, value


