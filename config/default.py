class RLConfig:

    def __init__(self, test_size=5, image_resolution=600, learning_rate=0.0007,  total_timesteps=100000, verbose=2,
                 folds=5, net_arch=None):

        self._net_arch = dict()

        self._test_size = test_size

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

        if int(folds) == folds:
            self._folds = folds
        else:
            self._folds = 5

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

    @property
    def folds(self):
        return self._folds

    @property
    def test_size(self):
        return self._test_size

    def __str__(self):
        return "RLConfig(test_size={}, image_resolution={}, learning_rate={}, total_timesteps={}, verbose={}, " \
               "folds={}, net_arch={})".format(self.test_size, self.image_resolution, self.learning_rate,
                                               self.total_timesteps, self.verbose, self.folds, self.net_arch)
