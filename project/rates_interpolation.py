import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

n=1000                                  # number interpolated values

input = pd.read_csv('input.csv', header=None)
input_matrix = np.array([input.loc[0, ], input.loc[1, ]], dtype='float64')

class rates_interpolation:
    interpolator = np.zeros((2, n))     # matrix of interpolated values and tenors

    def __init__(self, method, type, input_rates):
        self.m = method                 # which method
        self.t = type                   # type of rates
        self.ir = input_rates           # input rates

    def make_interpolator(self):
        if(self.m == "cubic_spline"):
            self.cubic_spline()
            self.draw_plot(self.m)
        if(self.m == "forward_monotone_convex_spline"):
            self.fmcs()
            self.draw_plot(self.m)

    def draw_plot(self, int_name):
        plt.plot(self.ir[1,], self.ir[0,], 'o')
        plt.plot(self.interpolator[1,], self.interpolator[0,])
        plt.xlabel("tenors")
        plt.ylabel("rates")
        plt.title(int_name)
        plt.legend(['draw', 'interpolation'])
        plt.show()

    def cubic_spline(self):
        func = interp1d(self.ir[1, ], self.ir[0, ], kind='cubic')
        time_points = np.linspace(self.ir[1, 0], self.ir[1, -1], n)
        self.interpolator = np.array([func(time_points), time_points])

    def fmcs(self):                      # Forward Monotone Convex Spline
        fwd = np.zeros(len(self.ir[0])+1, dtype='float64')
        fwd[1] = self.ir[1, 0]/self.ir[1, 1]*self.ir[0, 1] + (self.ir[1, 1] - self.ir[1, 0])/self.ir[1, 1]*self.ir[0, 0]
        for i in range(2, len(self.ir[0])):
            fwd[i] = (self.ir[1, i-1] - self.ir[1, i-2])/(self.ir[1, i] - self.ir[1, i-2])*self.ir[0, i] + \
                     (self.ir[1, i] - self.ir[1, i-1])/(self.ir[1, i] - self.ir[1, i-2])*self.ir[0, i-1]
        fwd[0] = self.ir[0, 0] - (fwd[1] - self.ir[0, 0])/2
        fwd[len(self.ir[0])] = self.ir[0, len(self.ir[0])-1] - (fwd[len(self.ir[0])-1] - self.ir[0, len(self.ir[0])-1])/2

        # The basic interpolator
        time_points = np.linspace(0, self.ir[1, -1], n)
        for i in range(0, n):
            index = np.searchsorted(self.ir[1, ], time_points[i], side='left')
            self.interpolator[0, i] = fwd[index] - (4*fwd[index] + 2*fwd[index + 1] - 6*self.ir[0, index])*self.x(time_points[i], index) + \
                                      (3*fwd[index] + 3*fwd[index + 1] - 6*self.ir[0, index])*self.x(time_points[i], index)*self.x(time_points[i], index)
            self.interpolator[1, i] = time_points[i]

    def x(self, tau, i):
        if(i==0):
            return (tau)/(self.ir[1, i])
        else:
            return (tau - self.ir[1, i-1])/(self.ir[1, i] - self.ir[1, i-1])

    def check(self):                # additional function - check the intergral of interpolation function
        time_points = np.linspace(0, self.ir[1, -1], n)
        sum = 0
        for i in range(1, n):
            index = np.searchsorted(self.ir[1, ], time_points[i], side='left')
            if(index == 1):
                sum = sum + self.interpolator[0, i]*(self.interpolator[1, i] - self.interpolator[1, i-1])
        return print(sum)

inter = rates_interpolation(input.loc[3, 0], input.loc[2, 0], input_matrix)
inter.make_interpolator()
inter.check()