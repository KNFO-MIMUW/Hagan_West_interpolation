import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pandas as pd

n=1000                       # number interpolated values
w =0.2
input = pd.read_csv('input.csv', header=None)
input_matrix = np.array([input.loc[0, ], input.loc[1, ]], dtype='float64')

class rates_interpolation:
    interpolator = np.zeros((2, n), dtype=np.complex128)     # matrix of interpolated values and tenors

    def __init__(self, method, type, input_rates):
        self.m = method                 # which method
        self.t = type                   # type of rates
        self.ir = input_rates           # input rates

    def make_interpolator(self):
        if(self.m == "cubic_spline"):
            self.cubic_spline()
        if(self.m == "forward_monotone_convex_spline"):
            self.fmcs()
        if(self.m == "raw"):
            self.raw()
        if(self.m == "minimalist"):
            self.minimalist()
        self.draw_plot(self.m)

    def draw_plot(self, int_name):
        plt.plot(self.ir[1,],self.ir[0,], 'o')
        plt.plot(self.interpolator[1,], self.interpolator[0,])
        plt.xlabel("tenors")
        plt.ylabel("rates")
        plt.title(int_name)
        plt.legend(['draw', 'interpolation'])
        plt.show()

    def cubic_spline(self):
        """ Cublic Spline for forward and spot curves 
        Input rates: only spot
        Output rates: spot or forward rates
        """
        # Compute Cublic Spline for spot rates
        func = CubicSpline(self.ir[1, ], self.ir[0, ])
        time_points = np.linspace(self.ir[1, 0], self.ir[1, -1], n)
        if self.t == "forward":
            # Calculate d/d(tau) r(tau)* tau = r'(tau) * tau + r(tau)
            func_derivative = func.derivative(nu=1)
            interpolation_values = func_derivative(time_points) * time_points + func(time_points)
            self.interpolator = np.array([interpolation_values, time_points])
        else: # spot rates 
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
        
    def raw(self):
        "Raw interpolation"
        time_points = np.linspace(self.ir[1,0], self.ir[1,-1], n)
        # Remove input nodes
        time_without_input = [i for i in time_points if i not in self.ir[1,:]] 
        self.interpolator = np.zeros((2, len(time_without_input)))
        self.interpolator[1,:] = time_without_input
        # Calculate  r(tau) = (r(i+1)tau(i+1) -r(i)tau(i)) /(tau(i+1) - tau(i)) for tau(i)< tau <tau(i+1)
        r_tau = self.ir[1,:] * self.ir[0,:] #r(i+1) * tau(i+1) vector
        tau_diff = np.diff(self.ir[1,:])  # Denominator of r(tau)
        raw_value = np.diff(r_tau) / tau_diff # Raw interpolation resluts
        # Update raw_interpolation for all time nodes (without input nodes)
        for i in range(0, self.ir.shape[1]-1):
            self.interpolator[0,np.where((self.interpolator[1,:] >self.ir[1,i])&(self.interpolator[1,:] < self.ir[1, i+1]))[0]] = raw_value[i]
        
    def minimalist(self):
        """The Minimalist Interpolation
        - moze tu jakis opis pozniej zrobie
        """
        #w = 0.8 # Moze warto to przerobic by mozma bylo sobie wyznaczyc samodzielnie
        # m - number of input nodes
        m = len(self.ir[0,:])
        tau = np.insert(self.ir[1,:],0,0)
        # h = tau(i) - tau(i-1)
        h = np.diff(tau)
        
        "Create vector v"
        # if 3m+1<=i<=4m v(i)=  f^d(i-3*m) 
        # otherwise v(i) = 0
        v = np.zeros(5 * m - 1)
        v[(3*m):(4*m)] = self.ir[0,:]
        
        "Create matrix A for solving the problem Az = v "
        A = np.zeros((5 * m -1, 5 * m - 1))
        
        #if i=1 :  -z(3m+1)-z(4m+1)=0
        A[0, [3 *m, 4*m]] =  [-1,-1]
        
        #if i=4,7,...,3m-5:  -z(3m +(i+2)/3) + z(4m-1 +(i+2)/3) - z(4m+(i+2)/3) =0
        for i in range(4, 3*m-2, 3):
            A[ i - 1, [3*m + (i+2)//3 - 1, 4 * m - 2 + (i+2)//3, 4*m - 1 + (i+2)//3]] = [-1, 1,-1]
            
        #if i=3m-2:  -z(4m)+z(5m-1)=0
        A[ 3*m - 3, [4*m - 1, 5*m -2]] = [-1, 1]  
        
        #if i=2:  -2w(z(5) -z(2) -2z(3)h1) - z(3m+1)*1/2 * h1 - z(4m+1)*h1)=0
        A[1, [4, 1, 2, 3*m, 4*m ]] = [ - 2*w, 2*w, 4*w*h[0], -h[0] /2, - h[0]]  
        
        #if i=5,8,...,3m-4:  2w(-z(i-3)-2(z-2)h((i-2)/3)+2z(i)+2z(i+1)h((i+1)/3)-z(i+3))
        #                   -z(3m+(i+1)/3)*(1/2)*h((i+1)/3) - z(4m+(i+1)/3) *h^2((i+1)/3)=0
        for i in range(5, 3*m -1 , 3):
            A[i-1, [i-4, i-3, i-1, i, i + 2]] = 2 * w * np.array([-1,-2 * h[(i-2)//3 - 1], 2, 2 *h[(i+1)//3 -1], -1])
            A[i-1, [3*m -1 +(i+1)//3, 4*m -1 + (i+1)//3]] = np.add(A[i-1, [3*m -1 +(i+1)//3, 4*m -1 + (i+1)//3]] ,[- 1/2 * h[(i+1)//3 -1], - h[(i+1)//3 -1] ** 2 ])
            
        #if i=3m-1:  2w(z(3m-1)-z(3m-4)-2z(3m-3)h(m-1)) -z(4m) * 1/2 =0
        A[3 *m - 2 , [3*m-2, 3*m-5, 3*m-4, 4*m -1]] = [2 *w , - 2*w, -4*w * h[m-2], - h[m-1]/2] 
        
        #if i=3,6,...,3m-3:  -4w(z(i+2)-z(i-1)-2z(i)h(i/3)) + 8(1-w)z(i)h^2(i/3) - z(3m+i/3)*1/3*h^2(i/3) - z(4m+i/3)h^2(i/3)
        for i in range(3, 3*m, 3):
            A[i-1, [i+1, i-2, i-1, 3*m-1 + i//3, 4*m-1 + i//3]] = [-4 * w, 4*w, 8*w*h[i//3 -1]+ 8*(1-w)*(h[(i//3)-1] ** 2), -1/3 * (h[i//3 - 1]**2),- (h[(i//3)-1] ** 2)]
        
        #if i=3m  8(1-w)z(3m)h^2(m) - z(4m)*1/3*(h(m))^2=0
        A[3*m-1, [3*m-1,4*m-1]] = [8*(1-w)*(h[m-1]**2), 1/3 * (h[m-1] ** 2)]
        
        #if 3m+1<=i<=4m:  z(3(i-3m)-2) + 1/2 * z(3(i-3m)-1)*h(i-3m) + 1/3 z(3(i-3m))*(h(i-3m))^2 = f^d(i-3m)
        for i in range(3*m +1, 4*m+1):
            A[i-1, [3*(i-3*m)-3, 3*(i-3*m)-2, 3*(i-3*m)-1]] = [1, 1/2 * h[i-1-3*m],1/3 * (h[i-1-3*m] ** 2)]
            
        #i4 4m+1<=i<=5n -1:  z(3(i-4m)-2)+z(3(i-4m)-1)h(i-4m)+z(3(i-4m))(h(i-4m))^2 -z(3(i-4m)+1)=0
        for i in range(4*m+1,5*m):
            A[i-1, [3*(i - 4 * m)-3, 3*(i - 4*m)-2, 3*(i-4*m)-1, 3*(i-4*m)]] = [1, h[i-4*m-1],(h[i-4*m-1]) ** 2,-1]

        " Solve linear system Az = v"
        # z = (z1,..., zn) = (a1,b1,c1,a2,...,an,bn,cn, lambda1, lambda2, ..., lambda(2n-1))
        z =np.linalg.solve(A,v)
        # x = (a1,b1,c1,..., an,bn,cn) - vector  minimalist interpolator coefficients
        x = z[0:(3*m)]
        a = x[0:len(x):3]
        b = x[1:len(x):3]
        c = x[2:len(x):3]
        #print(a + 1/2 * b * h + 1/3 * c * (h ** 2))
        " Find minimalist interpolator"
        #f(tau) = ai + bi * (tau - tau(i-1)) + ci(tau - tau(i-1))^2 for tau(i-1)<= tau <= tau(i)
        self.interpolator[1, :] = np.linspace(0, self.ir[1, -1],n)
        for i in range(0, self.ir.shape[1]):
            time_condition = np.where((self.interpolator[1, :] >=tau[i])&( self.interpolator[1, :] <= tau[i+1]))
            self.interpolator[0, time_condition] = a[i] + b[i] *(self.interpolator[1, time_condition] - tau[i]) + c[i] *((self.interpolator[1, time_condition] - tau[i]) ** 2)


inter = rates_interpolation(input.loc[3, 0], input.loc[2, 0], input_matrix)
inter = rates_interpolation("cubic_spline", "spot", input_matrix)
inter.make_interpolator()
