"""
This code simulates and solves the mother's problem and computes utility

"""

import numpy as np
import pandas as pd
import sys
import os
from scipy import stats
import math
from math import *
from scipy.optimize import minimize


class SimData:
    """

    """
    
    def __init__(self,N,model):
        """
        model: a utility instance (with arbitrary parameters)
        """
        self.N      = N
        self.model  = model
        
        #self.H, self.D = H, D
        
    def util(self, choice, shocks, wage):
        """
        This function takes labor and cc choices and computes utils
        """

        H = choice[0]
        D = choice[1]

        return self.model.utility(wage, shocks, H, D)

    
    def choice(self):
        """
        Computes optimal L^s and D_i values. Maximizes util(H, D).
        """
        zeros     = np.reshape(np.zeros(self.N), (self.N,1)) #(N,1)
        ones      = np.reshape(np.ones(self.N), (self.N,1))
       
        #l, d
        choice_0 = [zeros    , zeros] #(2, N)
        choice_1 = [zeros    , ones]
        choice_2 = [ones*160 , zeros]
        choice_3 = [ones*160 , ones]
        
        wage   = self.model.waget()
        shocks = self.model.res_causal()
        
        u_0 = self.util(choice_0, shocks, wage)
        u_1 = self.util(choice_1, shocks, wage)
        u_2 = self.util(choice_2, shocks, wage)
        u_3 = self.util(choice_3, shocks, wage)
        
        
        u = np.hstack([u_0, u_1, u_2, u_3])
        
        choice = np.argmax(u, axis=1)
        choice = np.reshape(choice, (self.N,1))
        #returns a (1,N) array
        
        dict= { 0: [0,0], 1: [0,1], 2: [160,0], 3: [160,1]}
        #según lo que entendí, se debe devolver el óptimo para cada individuo
        labor_opt = np.array(0)
        cc_opt    = np.array(0)

        
        for x in choice:
            x         = float(x)
            labor_opt = np.append(labor_opt , dict[x][0])
            cc_opt    = np.append(cc_opt    , dict[x][1])
        
        labor_opt = np.reshape(labor_opt[1:self.N+1], (self.N,1))
        cc_opt    = np.reshape(cc_opt[1:self.N+1], (self.N,1))
        score     = self.model.score(cc_opt, shocks)
        
    
        max_u = self.model.utility(shocks, wage, labor_opt, cc_opt)
        
        return {'Choice': choice,
                'Wage': wage,
                'Test Score': score,
                'Hours Choice': labor_opt,
                'CC Choice': cc_opt,
                'Max Utility': max_u}
