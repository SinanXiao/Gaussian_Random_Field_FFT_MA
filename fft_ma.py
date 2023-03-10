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

def fft_ma_3d(ny=50, dy=1, nx=50, dx=1, nz=50, dz=1, mean_value=0, stdev=1, scale=[20,2,2], angle=0):
    """
    simulating stationary Gaussian field over an 'ny'*'nx'*'nz' grid
    INPUT:   
                  ny: discretization size in 'y'
                  nx: discretization size in 'x'
                  xz: discretization size in 'z'
                  
                  dx: length of each cell in 'x'
                  dy: length of each cell in 'y'
                  dz: length of each cell in 'z'
                  
          mean_value: the mean of the Gaussian field, e.g., 0
               stdev: standard deviation, e.g., 1 
               scale: correlation lengths, e.g., [30,10,10] 
               angle: angle of rotation, e.g., [45,45,45] deg
               
    OUTPUT:  
             stationary Gaussian field in ny*nx*nz grid;
    """
    ndim = len(scale)
    
    nx_c = nextpow2(nx*2) # could use a larger number than 2
    ny_c = nextpow2(ny*2) # 
    nz_c = nextpow2(nz*2)
    
    # larger range
    x = np.arange(0, nx_c) * dx
    y = np.arange(0, ny_c) * dy
    z = np.arange(0, nz_c) * dz
    
    X, Y, Z = np.meshgrid(x,y,z)
    
    h_x = X - x[math.ceil(nx_c/2)]
    h_y = Y - y[math.ceil(ny_c/2)]
    h_z = Z - z[math.ceil(nz_c/2)]
    
    # coordinates of of all grids
    coords = np.stack((h_x.ravel(), h_y.ravel(), h_z.ravel()), axis=1)
    
    # covariance 
    cov = cal_cov(np.zeros(ndim), coords, stdev, scale, angle)
    cov = cov.reshape(ny_c, nx_c, nz_c)
    
    # FFT
    fftC = fft.fftn(fft.fftshift(cov))
    
    # normal deviates
    rng = np.random.default_rng()
    z_rand = rng.standard_normal(size=fftC.shape)
    
    # Invere FFT
    out = fft.ifftn(np.sqrt(fftC) * fft.fftn(z_rand))
    random_field = np.real(out[0:ny,0:nx,0:nz]) + mean_value
    
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
    ndim = pos2.shape[1]
    
    # difference between two positions
    dp = pos2 - pos1
    
    if ndim == 2:
        # ratation
        angle = angle * math.pi / 180
        RotMat = np.array([[math.cos(angle), -math.sin(angle)], \
                           [math.sin(angle),  math.cos(angle)]])
        dp = dp @ RotMat.T 
        
        # scale
        dp = dp/np.array(scale)
    
        # distance 
        dist = np.sqrt(dp[:,0]**2 + dp[:,1]**2)
        
    elif ndim == 3:
        if len(angle) != 3:
            #raise ValueError("angle must have 3 element")
            print(" 'angle' doesn't have 3 elements")
            print(" use angle = [0,0,0]")
            angle = [0,0,0]
        
        # ratation 
        angle = np.array(angle) * math.pi / 180
        
        T1 = np.array([[1,                  0,                   0], \
                       [0, math.cos(angle[2]), -math.sin(angle[2])], \
                       [0, math.sin(angle[2]),  math.cos(angle[2])]])
        T2 = np.array([[ math.cos(angle[1]), 0, math.sin(angle[1])], \
                       [0,                1,                     0], \
                       [-math.sin(angle[1]), 0, math.cos(angle[1])]])
        T3 = np.array([[math.cos(angle[0]), -math.sin(angle[0]), 0], \
                       [math.sin(angle[0]),  math.cos(angle[0]), 0], \
                       [0,               0,                      1]])
        
        RotMat = T1@T2@T3
        dp = dp @ RotMat.T 
        
        # scale
        dp = dp/np.array(scale)
    
        # distance 
        dist = np.sqrt(np.sum(dp**2,axis=1))
        
    # covariance
    semiv = semi_variogram(dist, stdev)
    cov = stdev**2 - semiv
    
    return cov
    
def semi_variogram(h, stdev):
    
    semiv = stdev**2 * (1 - np.exp(-h**2)) # Gaussian
    
    return semiv
    
if __name__ == "__main__":
    
    # 2D
    random_field = fft_ma_2d(nx=100, ny=100, scale=[30,3], angle=0)
    plt.figure()
    plt.pcolor(random_field)
    plt.show()

    # 3D
    random_field = fft_ma_3d(nx=50, ny=50, nz=50, scale=[20,2,5], angle=[0,0,0])
    for i in range(0,30,10):
        plt.figure()
        plt.pcolor(random_field[:,:,i])
        plt.show()