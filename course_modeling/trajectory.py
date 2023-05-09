"""trajectory.py - Trajectory Factories"""
import numpy as np
import scipy.optimize as opt

import matplotlib.pyplot as plt

import os, sys

sys.path.insert(0, os.path.realpath('.'))
from course_modeling.course import Course

# %% Interpolants
def cross(u: np.ndarray, v: np.ndarray) -> float:
    """Compute 2D cross-product"""
    return u[0]*v[1] - u[1]*v[0]

def planar_cubic_g2_interpolant(T: np.ndarray, bc: str = 'natural', 
                                alpha: float = 0.5, eps: float = 1e-3):
    """Generates a planar cubic G^2 continuous interpolant spline. Refer to:
    'Interpolation Scheme for Planar Cubic G^2 Spline Curves' by Krajnc. 
    """    
    m = T.shape[-1] - 1

    # Add terminal points according to boundary conditions
    match bc:
        case 'natural':
            T = np.insert(T, 0, (T[:,[0]] - np.diff(T[:,:2])).T, axis=1)
            T = np.append(T, T[:,[-1]] + np.diff(T[:,-2:]), axis=1)
        case 'periodic':
            T = np.insert(T, 0, T[:,-1], axis=1)
            T = np.append(T, T[:,[0]], axis=1)

    # Difference data
    dT = np.diff(T).T
    dT_norm = np.array([np.linalg.norm(dT_l) for dT_l in dT]) 

    # Quadratic direction approximation
    u = np.empty((m+1,))
    d = np.empty((m+1,2))
    for l in range(m+1):
        u[l] = dT_norm[l]**alpha / (dT_norm[l]**alpha + dT_norm[l+1]**alpha)

        d[l] = ((1 - u[l])/u[l]) * dT[l,:] + (u[l]/(1 - u[l])) * dT[l+1,:]
        d[l] = d[l] / np.linalg.norm(d[l])
    
    # Segment classification
    delta = np.empty((m,3))
    S = np.empty((m,))
    K = np.empty((m,2))
    for l in range(m):
        delta[l,0] = cross(d[l,:] , dT[l,:] )
        delta[l,1] = cross(dT[l,:], d[l+1,:])
        delta[l,2] = cross(d[l,:] , d[l+1,:])

        D02 = delta[l,0] * delta[l,2] < 0
        D12 = delta[l,1] * delta[l,2] < 0

        if np.any(delta[l,:] == 0):
            S[l] = 0        
        else:
            S[l] = 1 + D02 + 2*D12

        K[l,0] = 2/3*np.abs(delta[l,0]) * (delta[l,2] / delta[l,1])**2
        K[l,1] = 2/3*np.abs(delta[l,1]) * (delta[l,2] / delta[l,0])**2

    # Curvature sign
    Delta = np.empty((m+1,))
    for l in range(m+1):
        Delta[l] = cross(dT[l,:], dT[l+1,:])
    
    # Curvature lower bound
    B = np.empty((m+1,))

    B[0]  = K[0 ,0] if S[0]  in [1,2] else 0
    B[-1] = K[-1,1] if S[-1] in [1,3] else 0
    
    for l in range(1,m-1):
        if S[l-1] in [1,3]:
            if S[l] in [1,2]:
                B[l] = np.max([K[l-1,1], K[l,0]])
            else:
                B[l] = K[l-1,1]
        else:
            if S[l] in [1,2]:
                B[l] = K[l,0]
            else:
                B[l] = 0
        
    # Quadratic curvature approximation
    v = np.empty((m+1,))
    for l in range(m+1):
        v[l] = (2 * Delta[l] * u[l]**2 * (1-u[l])**2) \
            / (
                (1-u[l])**4 * dT_norm[l]
                + 2 * u[l]**2 * (1-u[l])**2 * np.dot(dT[l,:], dT[l+1,:])
                + u[l]**4 * dT_norm[l+1]
            )**(3/2)
    
    # Set knot curvatures
    kappa = np.empty((m+1,))
    for l in range(m+1):
        if v[l] > B[l]:
            kappa[l] = np.sign(Delta[l])*v[l]
        else:
            kappa[l] = np.sign(Delta[l])*(B[l] + eps)
    
    # Compute tensions   
    eta = np.empty((m, 2))      # Denoted by lambda in paper
    for l in range(m):
        R0 = 3/2 * kappa[l]/delta[l,0] * (delta[l,1]/delta[l,2])**2
        R1 = 3/2 * kappa[l]/delta[l,1] * (delta[l,0]/delta[l,2])**2

        roots = np.roots([(R0**2)*R1, 0, -2*R0*R1, 1, R1-1])

        rho0 = roots
        rho1 = 1 - R0 * rho0**2  

        eta[l,0] = 3 * rho0 * delta[l,1] / delta[l,2]
        eta[l,1] = 3 * rho1 * delta[l,0] / delta[l,2]

    return d, eta

def main():
    T = np.array([[0, 1, 2, 2.5, 1.5, 2, 1, -1],
                  [0, 1, 0, 2  , 2  , 3, 4,  3]])
    
    d, eta = planar_cubic_g2_interpolant(T)

    for l in range(T.shape[-1]):
        _pause = True

if __name__ == '__main__':
    main()