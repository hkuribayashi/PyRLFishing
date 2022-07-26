import os


class RLConfig():

    def __init__(self, image_resolution, learning_rate, net_arch, total_timesteps, verbose):
        if image_resolution is None or not isinstance(image_resolution, int):
            raise RuntimeError('The parameter image_resolution should be a int value: {}'.format(image_resolution))
        else:
            self._image_resolution = image_resolution

        if learning_rate is None or not isinstance(learning_rate, float):
            raise RuntimeError('The parameter learning_rate should be a float value: {}'.format(learning_rate))
        else:
            self._learning_rate = learning_rate

        if net_arch is None:
            raise RuntimeError('The parameter net_arch should be an array value: {}'.format(net_arch))
        else:
            self._net_arch = net_arch

        if total_timesteps is None or not isinstance(total_timesteps, int):
            raise RuntimeError('The parameter total_timesteps should be a int value: {}'.format(total_timesteps))
        else:
            self._total_timesteps = total_timesteps

        if verbose is None or not isinstance(verbose, int):
            raise RuntimeError('The parameter verbose should be an int value: {}'.format(verbose))
        else:
            self._verbose = verbose

    @property
    def image_resolution(self):
        return self._image_resolution

    @property
    def learning_rate(self):
        return self._image_resolution

    @property
    def net_arch(self):
        return self._net_arch

    @property
    def total_timesteps(self):
        return self._total_timesteps

    @property
    def verbose(self):
        return self._verbose
