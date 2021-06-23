class Quad(object):

    def __init__(self, x_image, y_image, mag, t_arrival=None):
        """
        Data class for a quadruply-imaged quasar

        :param x_image: x image positions
        :param y_image: y image positions
        :param mag: image magnifications or flux ratios
        :param t_arrvival: arrival times (optional)
        """

        self.x, self.y = x_image, y_image
        self.m = mag
        self._nimg = len(x_image)
        self.t_arrival = t_arrival
