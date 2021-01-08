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
        self.param, self.N, self.data = param, N, data


    def waget(self):
        """
        Computes w (hourly wage) for periodt

        lnw = beta0 + beta1*esc + e

        where e iid normal
        
        Returns wage as a column vector
        """

        epsilon = np.sqrt(self.param.sigma2w)*np.random.randn(self.N)
                
        xw = np.concatenate((np.reshape(np.array(self.data['constant'], dtype=np.float64),(self.N,1)),
                             np.reshape(np.array(self.data['m_sch'], dtype=np.float64), (self.N,1)),), axis=1)
        
        betas = self.param.betasw
        

        return np.reshape(np.exp(np.dot(xw,betas)+ epsilon),(self.N,1))


    def res_causal(self):
        """
     
        Computes \theta y \omega from a bivariate normal distribution using a given mean and covariance matrix
        
        Returns array with objects omega [0] and theta [1]
        """
        
        s = np.random.multivariate_normal(self.param.meanshocks, self.param.covshocks, self.N)
        omega = np.reshape(s[:,0],(self.N,1))
        theta = np.reshape(s[:,1],(self.N,1))
      
        return omega, theta


    def score(self, cc_choice, shocks):
        """
        Computes cognitive test score 
        
        Sc = beta_0+theta*d_cc_34+u
        """
        u = np.reshape(np.random.randn(self.N), (self.N,1))
        ones = np.reshape(np.ones(self.N), (self.N,1))

        return  self.param.betast[0]*ones + np.multiply(shocks[1], cc_choice) + u
     
    
    
    def nli(self, labor_choice):
        """
        Computes non-labor income 
        
        d_married : dummy that equals 1 when mother is married; 0 otherwise
        d_work    : dummy that equals 1 when mother chooses to work; 0 otherwise
        n_kids    : number of children
        sch       : years of schooling
        
        ln_nli = beta_0+beta_1*d_married+beta_2*d_work+beta_3*n_kids+beta_4*sch+w
        """
        
        epsilon = np.random.randn(self.N)
        xn = np.concatenate((np.reshape(np.array(self.data['constant'], dtype=np.float64),(self.N,1)),
                             np.reshape(np.array(self.data['married34'], dtype=np.float64), (self.N,1)),
                             np.reshape(np.array(labor_choice, dtype=np.float64), (self.N,1)),
                             np.reshape(np.array(self.data['tot_kids'], dtype=np.float64), (self.N,1)),
                             np.reshape(np.array(self.data['m_sch'], dtype=np.float64), (self.N,1)),), axis=1)
      
        betas = self.param.betasn
        
        return np.reshape(np.exp(np.dot(xn,betas)+ epsilon),(self.N,1))

    
    def utility(self, wage, shocks, H, D):

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
        
        Income: 
        income = labor income + non labor income = wage*hours + nli
        
        Returns column vector of Utility, computed as:
            U = ln(wage*H) + alpha*L + D*nu

        """
        d_work = H/160
        nli = self.nli(d_work)
        income = wage*H + nli
        
        C = D*np.reshape(np.array(self.data['commute2cc'],dtype=np.float64), (self.N, 1)) #commute*dummy cc
        
        alpha  = self.param.alpha
        gamma  = self.param.gamma
        
        nu =  -shocks[0] + gamma*shocks[1]
        
        L = self.param.T - (1-D)*self.param.Lc - H - 2*C
        

        return np.log(income) + L*alpha + D*nu






