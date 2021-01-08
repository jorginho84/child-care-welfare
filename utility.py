"""


-Utility class: takes parameters, X's, and given choices
computes utility


"""
# from __future__ import division #omit for python 3.x
import numpy as np
import pandas as pd
import sys
import os
from scipy import stats
import math
from math import *


class Utility(object):
    """ 

    This class defines the economic environment of the agent

    """

    def __init__(self, param, N, data):
        """
        Set up model's data and parameters

        """
        self.N, self.param, self.data = N, param, data


    def waget(self):
        """
        Computes w (hourly wage) for periodt

        lnw = beta0 + beta1*esc + e

        where e iid normal
        
        Returns wage as a column vector
        """

        epsilon = np.sqrt(self.param.sigmaw)*np.random.randn(self.N)
        
        #constant is always the last column:
        colbeta0 = len(self.data[0])-1

        xw = np.concatenate((np.reshape(self.data[:,0], (self.N,1)), #escolaridad
                            np.reshape(self.data[:,colbeta0], (self.N,1)),),axis=1) #constante
        
        betas = self.param.betas
        

        return np.reshape(np.exp(np.dot(xw,betas)+ epsilon),(self.N,1))


    def res_causal(self):
        """
     
        Computes \theta y \omega from a bivariate normal distribution using a given mean and covariance matrix
        
        Returns array with objects omega (0) and theta (1)
        """
        
        s = np.random.multivariate_normal(self.param.meanshocks, self.param.covshocks, self.N)
        omega = np.reshape(s[:,0],(self.N,1))
        theta = np.reshape(s[:,1],(self.N,1))
      
        return omega, theta


    def utility(self, shocks, wage, H, D):
        """
        
        L_i     : "leisure" understood as time not working, sleeping, commuting or spent in cc
        H_i     : hours worked (0,40)
        D_i     : dummy child care
        delta_i : commute to closest cc center (hours) 
        
        T_i - (1-D_i)*Lc_i = L_i + H_i + D_i*delta_i
        T_i - (1-D_i)*Lc_i = L_i + H_i + C_i
        
        shocks  : 2-element array, omega=shocks[0], theta=shocks[1]; 
        wage    : hourly wage (monthly wage/(4*40))
        
        Computes nu: 
        nu = -omega + gamma*theta
        
        Returns column vector of Utility, computed as:
            U = ln(wage*H) + alpha*L + D*nu

        """
        
        C = D*np.reshape(self.data[:,1], (self.N, 1)) #commute*dummy cc
        
        alpha  = self.param.alpha
        gamma  = self.param.gamma
        
        nu =  -shocks[0] + gamma*shocks[1]
        
        L = self.param.T - (1-D)*self.param.Lc - H - 2*C
        
        return np.log(wage*H + sys.float_info.epsilon) + L*alpha + D*nu







