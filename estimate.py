#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:47:22 2020

@author: akac
"""

import numpy as np
import pandas as pd
import pickle
import tracemalloc
import itertools
import sys, os
import statsmodels.api as sm
from scipy import stats
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sn
from statsmodels.iolib.summary2 import summary_col




class estimate:
    "This class "
    
    def __init__(self, N, data, param, model, model_sim):
        "Initial class"
        
        self.N, self.data, self.param = N, data, param
        self.model, self.model_sim    = model, model_sim
        
    
    def simulation(self,times):
        """
        Function that simulates x times.
        
        Variables:
            Yi: cognitive test score (at 5-7 years of age)
            Di: child care dummy
            Zi : distancia (en tiempo)
            
        Estimation 1: Y_i = alpha0 + alpha1*D_i  + u_i
        Estimation 2: D_i = gamma0 + gamma1*Zi   + v_i
        Estimation 3: Y_i = beta0 + beta1*Zi     + w_i

        1.	Varianza del error de la ecuación de salarios
        2.	Proporcion de niños en child care
        3.	Estimación OLS de alpha 1 en: Y_i = alpha0 + alpha1*D_i  + error.
        4.	Estimacion OLS gamma1 en: Di = gamma0 + gamma1*Zi + error.
        5.	Varianza de Yi.
        6.	Estimación OLS de beta1 en: Yi = beta0 + beta1*Zi + error.

        La misma idea que hemos trabajado:
            – Simular test scores, child care dummy. Usar misma variable de distancia.
            – Correr regresiones con data simulada, dentro del loop en estimate.py. Obtener promedio de estas estimaciones
            – Correr regresiones con data real. Calcular SEs de cada uno de estos momentos usando bootstrap
            – Agregar momentos (simulados, data y SE) en la tabla.
        """
        alpha1_est = np.zeros(times)
        beta1_est  = np.zeros(times)
        gamma1_est = np.zeros(times)
        
        b0_est     = np.zeros(times)
        b1_est     = np.zeros(times)
        varres_est = np.zeros(times)
        
        mean_labor = np.zeros(times)
        mean_cc    = np.zeros(times)
        mean_score = np.zeros(times)
        
        var_score  = np.zeros(times)
    
        
        
        for i in range(0,times):
            
            sim = self.model_sim.choice()
            
            ln_w = np.log(sim['Wage'])
            y = pd.DataFrame(ln_w)
            score = pd.DataFrame(sim['Test Score'])
                        
            
            if sim['Choice'][i]==2 or sim['Choice'][i]==3:  
                
                "log_w = beta_0 + beta_1*m_sch + e_i"
                reg_w         = sm.OLS(endog = y, exog=self.data[['constant', 'm_sch']], missing='drop').fit()
                b0_est[i]     = reg_w.params[0]
                b1_est[i]     = reg_w.params[1]
                varres_est[i] = np.var(reg_w.resid)
            
            "Y_i = alpha_0 + alpha_1*D_i + u_i"
            reg_1         = sm.OLS(endog = score, exog=self.data[['constant', 'd_cc_34']], missing='drop').fit()
            alpha1_est[i] = reg_1.params[1]
            
            "D_i = gamma_0 + gamma_1*Z_i + v_i"
            reg_2         = sm.OLS(endog=self.data['d_cc_34'], exog=self.data[['constant', 'commute2cc']], missing='drop').fit()
            gamma1_est[i] = reg_2.params[1]
            
            "Y_i=beta_0+beta_1*Z_i + w_i"
            reg_3         = sm.OLS(endog = score, exog=self.data[['constant', 'commute2cc']], missing='drop').fit()
            beta1_est[i]  = reg_3.params[1] 
            
            mean_labor[i] = np.nanmean(sim['Hours Choice']/160) 
            mean_cc[i]    = np.nanmean(sim['CC Choice'])
            mean_score[i] = np.nanmean(sim['Test Score'])
            var_score[i]  = np.var(sim['Test Score'])
            
        
        alpha1_sim = np.mean(alpha1_est)
        beta1_sim  = np.mean(beta1_est)
        gamma1_sim = np.mean(gamma1_est)
        
        b0_sim     = np.mean(b0_est) 
        b1_sim     = np.mean(b1_est) 
        varres_sim = np.mean(varres_est)
        
        labor_sim  = np.mean(mean_labor)
        cc_sim     = np.mean(mean_cc)
        score_sim  = np.mean(mean_score)
        
        var_score_sim = np.mean(var_score)
        
        return { 'Labor Choice': labor_sim,
                'CC Choice': cc_sim,
                'Test Score': score_sim,
                'Beta0': b0_sim,
                'Beta1': b1_sim,
                'Resid var': varres_sim,
                'alpha_1': alpha1_sim,
                'beta_1': beta1_sim,
                'gamma_1': gamma1_sim,
                'Var Score': var_score_sim}
    
        









