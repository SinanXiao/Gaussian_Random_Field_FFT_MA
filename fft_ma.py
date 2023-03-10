import numpy as np
import math
from numpy import fft
import matplotlib.pyplot as plt

def fft_ma_2d(ny=100, dy=1, nx=100, dx=1, mean_value=0, stdev=1, scale=[30,3], angle=0):
    """
    simulating stationary Gaussian field over an 'ny' times 'nx' grid
    INPUT:   
                  ny: discretization size in 'y'
                  nx: discretization size in 'x'
                  
                  dx: length of each cell in 'x'
                  dy: length of each cell in 'y'
                  
                  
          mean_value: the mean of the Gaussian field, e.g., 0
               stdev: standard deviation, e.g., 1 
               scale: correlation lengths, e.g., [30,10] 
               angle: angle of rotation, e.g., 45 deg
               
    OUTPUT:  
             stationary Gaussian field in ny*nx grid;
    """
    
    #x = np.arange(0, nx*dx, dx)
    #y = np.arange(0, ny*dy, dy)
    #z = np.arange(0, nz*dz, dz)
    
    ndim = len(scale)
    
    nx_c = nextpow2(nx*2) # could use a larger number than 2
    ny_c = nextpow2(ny*2) # 
    
    # larger range
    x = np.arange(0, nx_c) * dx
    y = np.arange(0, ny_c) * dy
    
    X, Y = np.meshgrid(x,y)
    
    h_x = X - x[math.ceil(nx_c/2)]
    h_y = Y - y[math.ceil(ny_c/2)]
    
    # coordinates of of all grids
    coords = np.stack((h_x.ravel(), h_y.ravel()), axis=1)
    
    # covariance 
    cov = cal_cov(np.zeros(ndim), coords, stdev, scale, angle)
    cov = cov.reshape(ny_c, nx_c)
    
    # FFT
    fftC = fft.fft2(fft.fftshift(cov))
    
    # normal deviates
    rng = np.random.default_rng()
    z_rand = rng.standard_normal(size=fftC.shape)
    
    # Invere FFT
    out = fft.ifft2(np.sqrt(fftC) * fft.fft2(z_rand))
    random_field = np.real(out[0:ny, 0:nx]) + mean_value
    
    return random_field


def nextpow2(x):
    """
    next higher power of 2
    """
    return 1 if x == 0 else 2**math.ceil(math.log2(x))
    
    
def cal_cov(pos1, pos2, stdev, scale, angle):
    """
    calculate covariance matrix
    """
    
    # difference between two positions
    dp = pos2 - pos1
    
    # ratation
    angle = angle * math.pi / 180
    RotMat = np.array([[math.cos(angle), -math.sin(angle)], \
                       [math.sin(angle),  math.cos(angle)]])
    dp = dp @ RotMat.T 
    
    # scale
    dp = dp/np.array(scale)
    
    # distance 
    dist = np.sqrt(dp[:,0]**2 + dp[:,1]**2)
    
    # covariance
    semiv = semi_variogram(dist, stdev)
    cov = stdev**2 - semiv
    
    return cov
    
def semi_variogram(h, stdev):
    
    semiv = stdev**2 * (1 - np.exp(-h**2)) # Gaussian
    
    return semiv
    
if __name__ == "__main__":
    from time import time
    start = time()
    random_field = fft_ma_2d(nx=100, ny=100, scale=[30,3], angle=0)
    print(time() - start)
    plt.figure()
    plt.pcolor(random_field)
    plt.colorbar()
    plt.show()
