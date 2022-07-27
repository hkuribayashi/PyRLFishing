class RLConfig:

    def __init__(self, image_resolution=600, learning_rate=0.0007,  total_timesteps=100000, verbose=2, net_arch=None):
        if int(image_resolution) == image_resolution:
            self._image_resolution = image_resolution
        else:
            self._image_resolution = 600

        if float(learning_rate) == learning_rate:
            self._learning_rate = learning_rate
        else:
            self._learning_rate = 0.0007

        if net_arch is None:
            self._net_arch = [64, 64]
        else:
            self._net_arch = net_arch

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
