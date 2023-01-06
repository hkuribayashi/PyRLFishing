class RLConfig:

    def __init__(self, image_resolution=600, learning_rate=0.0007,  total_timesteps=100000, verbose=2, net_arch=None):

        self._net_arch = dict()

        if int(image_resolution) == image_resolution:
            self._image_resolution = image_resolution
        else:
            self._image_resolution = 600

        if float(learning_rate) == learning_rate:
            self._learning_rate = learning_rate
        else:
            self._learning_rate = 0.0007

        if type(net_arch) == int:
            if net_arch == 64:
                self._net_arch['DQN'] = [64, 64]
                self._net_arch['A2C'] = [{'pi': [64, 64], 'vf': [64, 64]}]
                self._net_arch['PPO'] = [{'pi': [64, 64], 'vf': [64, 64]}]
            elif net_arch == 128:
                self._net_arch['DQN'] = [128, 128]
                self._net_arch['A2C'] = [{'pi': [128, 128], 'vf': [128, 128]}]
                self._net_arch['PPO'] = [{'pi': [128, 128], 'vf': [128, 128]}]
            elif net_arch == 256:
                self._net_arch['DQN'] = [256, 256]
                self._net_arch['A2C'] = [{'pi': [256, 256], 'vf': [256, 256]}]
                self._net_arch['PPO'] = [{'pi': [256, 256], 'vf': [256, 256]}]
            else:
                raise Exception("Valor de net_arch invalido: {}".format(net_arch))
        else:
            self._net_arch['DQN'] = [64, 64]
            self._net_arch['A2C'] = [{'pi': [64, 64], 'vf': [64, 64]}]
            self._net_arch['PPO'] = [{'pi': [64, 64], 'vf': [64, 64]}]

        if int(total_timesteps) == total_timesteps:
            self._total_timesteps = total_timesteps
        else:
            self._total_timesteps = 100000

        if int(verbose) == verbose:
            self._verbose = verbose
        else:
            self._verbose = 2

    def __str__(self):
        return "RLConfig (image_resolution={}, learning_rate={}, total_timesteps={}, verbose={}, net_arch={})".format(
            self.image_resolution, self.learning_rate, self.total_timesteps, self.verbose, self.net_arch)

    @property
    def image_resolution(self):
        return self._image_resolution

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def net_arch(self):
        return self._net_arch

    @property
    def total_timesteps(self):
        return self._total_timesteps

    @property
    def verbose(self):
        return self._verbose
