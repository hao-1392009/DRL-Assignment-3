import abc


class Agent(abc.ABC):
    def __init__(self, config):
        self.batch_size = config["batch_size"]
        self.device = config["device"]
        self.gamma = config["gamma"]
        self.n_step_td = config["n_step_td"]
        self.is_training = False

        self.target_network = None
        self.online_network = None
        self.replay_buffer = None

    def start_training(self):
        self.is_training = True

    @abc.abstractmethod
    def get_action(self, state):
        pass

    @abc.abstractmethod
    def update_online(self):
        pass

    def update_target(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    @abc.abstractmethod
    def save(self, output_dir):
        pass

    def on_episode_end(self, episode):
        pass
