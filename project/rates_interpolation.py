import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

n=1000                                  # number interpolated values

input = pd.read_csv('input.csv', header=None)
input_matrix = np.array([input.loc[0, ], input.loc[1, ]], dtype='float64')
input_matrix[0,0] = float(input_matrix[0,0])
input_matrix[1,0] = float(input_matrix[1,0])

class rates_interpolation:
    interpolator = np.zeros((2, n))     # matrix of interpolated values and tenors

    def __init__(self, m, t, ir):
        self.method = m                 # which method
        self.type = t                   # type of rates
        self.input_rates = ir           # input rates

    def make_interpolator(self):
        if(self.method == "cubic_spline"):
            self.cubic_spline()
            self.draw_plot()

    def draw_plot(self):
        plt.plot(self.input_rates[1,], self.input_rates[0,], 'o')
        plt.plot(self.interpolator[1,], self.interpolator[0,])
        plt.xlabel("tenors")
        plt.ylabel("rates")
        plt.title("interpolation")
        plt.legend(['draw', 'interpolation'])
        plt.show()

    def cubic_spline(self):
        f = interp1d(self.input_rates[1, ], self.input_rates[0, ], kind='cubic')
        time_points = np.linspace(self.input_rates[1, 0], self.input_rates[1, -1], n)
        self.interpolator = np.array([f(time_points), time_points])

inter = rates_interpolation(input.loc[3, 0], input.loc[2, 0], input_matrix)
inter.make_interpolator()