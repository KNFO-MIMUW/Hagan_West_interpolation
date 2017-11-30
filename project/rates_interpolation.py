import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

n=1000                                  # number interpolated values

class rates_interpolation:
    interpolator = np.zeros((2, n))     # matrix of interpolated values and tenors

    def __init__(self, m, t, ir):
        self.method = m                 # which method
        self.type = t                   # type of rates
        self.input_rates = ir           # input rates

    def make_interpolator(self):
        if(self.method == "CS"):
            self.CS()
            self.draw_plot()

    def draw_plot(self):
        plt.plot(self.input_rates[1,], self.input_rates[0,], 'o')
        plt.plot(self.interpolator[1,], self.interpolator[0,])
        plt.xlabel("tenors")
        plt.ylabel("rates")
        plt.title("interpolation")
        plt.legend(['draw', 'interpolation'])
        plt.show()

    def CS(self):
        f = interp1d(self.input_rates[1, ], self.input_rates[0, ], kind='cubic')
        time_points = np.linspace(self.input_rates[1, 0], self.input_rates[1, -1], n)
        self.interpolator = np.array([f(time_points), time_points])

matrix = np.array([[0.01, 0.02, 0.03, 0.02, 0.05], [1, 2, 3, 4, 5]])
inter = rates_interpolation("CS", "forwards", matrix)
inter.CS()
inter.make_interpolator()