#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:47:22 2020

@author: akac
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

import utility    as util
#import parameters as parameters
import simdata    as simdata


class estimate:
    "This class "
    
    def __init__(self, N, data, param, moments_boot, w_matrix):
        "Initial class"
        
        self.N, self.data, self.param = N, data, param
        self.moments_boot, self.w_matrix = moments_boot, w_matrix

        
    
    def simulation(self, model_sim):
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
        times = self.param.times
        b1_td_est  = np.zeros(times)
        b1_tz_est  = np.zeros(times)
        b1_dz_est  = np.zeros(times)
        
        b0_est     = np.zeros(times)
        b1_est     = np.zeros(times)
        varres_est = np.zeros(times)
        
        mean_labor = np.zeros(times)
        mean_cc    = np.zeros(times)
        mean_score = np.zeros(times)
        
        var_score  = np.zeros(times)
    
        
        
        for i in range(0,times):
            np.random.seed(i)
            
            sim = model_sim.choice()
            
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
            d = {'constant': np.ones(ln_w.size),'CC':sim['CC Choice'][:,0]}
            reg_1         = sm.OLS(endog = score, exog=pd.DataFrame(data=d), missing='drop').fit()
            b1_td_est[i]  = reg_1.params[1]
            
            "D_i = gamma_0 + gamma_1*Z_i + v_i"
            reg_2         = sm.OLS(endog=self.data['d_cc_34'], exog=self.data[['constant', 'commute2cc']], missing='drop').fit()
            b1_dz_est[i]  = reg_2.params[1]
            
            "Y_i=beta_0+beta_1*Z_i + w_i"
            reg_3         = sm.OLS(endog = score, exog=self.data[['constant', 'commute2cc']], missing='drop').fit()
            b1_tz_est[i]  = reg_3.params[1] 
            
            mean_labor[i] = np.nanmean(sim['Hours Choice']/160) 
            mean_cc[i]    = np.nanmean(sim['CC Choice'])
            mean_score[i] = np.nanmean(sim['Test Score'])
            var_score[i]  = np.var(sim['Test Score'])
            
        
        b1_td_sim = np.mean(b1_td_est)
        b1_tz_sim = np.mean(b1_tz_est)
        b1_dz_sim = np.mean(b1_dz_est)#cambiar nombre 
        
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
                'beta0_w': b0_sim,
                'beta1_w': b1_sim,
                'resid_var_w': varres_sim,
                'beta1_td': b1_td_sim,
                'beta1_tz': b1_tz_sim,
                'beta1_dz': b1_dz_sim,
                'resid_var_td': var_score_sim}
    
    
    def ll(self, beta):
        """
		Takes structural parameters and computes the objective function for optimization 
        
        beta = [beta]
		
		"""
		#updating beta->parameters instance to compute likelihood.
     
        self.param.alpha           = beta[0] #disutility from work
        self.param.gamma           = beta[1] #resistance to treatment
        self.param.meanshocks[0]   = beta[2] #mean of resistance to treatment
        self.param.meanshocks[1]   = beta[3] #mean of theta (causal effect of cc on child skills )
        self.param.covshocks[1]    = beta[4] #var of causal effect
        self.param.covshocks[2]    = beta[5] #correlation
        self.param.betasw[0]        = beta[6] #structural parameter of wage equation
        self.param.betasw[1]        = beta[7] #structural parameter of wage equation
        self.param.sigma2w_reg    = beta[8] #variance of res of wage equation
        self.param.betastd      = beta[9] #constant from test score equation
        
    

        model  = util.Utility(self.param, self.N, self.data)
        
        model_sim = simdata.SimData(self.N, model)
        
        est = self.simulation(model_sim)
                
        labor_choice = est['Labor Choice']
        cc_choice    = est['CC Choice']
        test_score   = est['Test Score']
        beta0_w      = est['beta0_w']
        beta1_w      = est['beta1_w']
        resid_var_w  = est['resid_var_w']
        beta1_td     = est['beta1_td']
        beta1_tz     = est['beta1_tz']
        beta1_dz     = est['beta1_dz']
        resid_var_td = est['resid_var_td']
            
        
        #number of moments to match
        num_par = labor_choice.size + cc_choice.size + test_score.size + beta0_w.size + beta1_w.size +resid_var_w.size + beta1_td.size + beta1_tz.size + beta1_dz.size + resid_var_td.size
        
        
        #outer matrix
        x_vector=np.zeros((num_par,1))
        
        #10 momentos 
        x_vector[0,0] = labor_choice - self.moments_boot['Labor Choice']
        x_vector[1,0] = cc_choice    - self.moments_boot['CC Choice']
        x_vector[2,0] = test_score   - self.moments_boot['Test Score']
        x_vector[3,0] = beta0_w      - self.moments_boot['beta0_w'] #beta0 from wage
        x_vector[4,0] = beta1_w      - self.moments_boot['beta1_w'] #beta1 from wage
        x_vector[5,0] = resid_var_w  - self.moments_boot['resid_var_w'] #resid var from wage
        x_vector[6,0] = beta1_td     - self.moments_boot['beta1_td'] #beta1 from test vs d_cc (td)
        x_vector[7,0] = beta1_dz     - self.moments_boot['beta1_dz'] #beta1 from d_cc vs commute (dz)
        x_vector[8,0] = beta1_tz     - self.moments_boot['beta1_tz'] #beta1 from test vs commute (tz)
        x_vector[9,0] = resid_var_td - self.moments_boot['resid_var_td'] #resid var from t vs d
        
        
        #The Q metric
        q_w = np.dot(np.dot(np.transpose(x_vector),self.w_matrix),x_vector)
        print ('')
        print ('The objetive function value equals ', q_w)
        print ('')

        return q_w


		
    def optimizer(self):
        
        beta0 = np.array([self.param.alpha,
                          self.param.gamma,
                          self.param.meanshocks[0],
                          self.param.meanshocks[1],
                          self.param.covshocks[1],#variance of causal effect
                          self.param.covshocks[2],#correlation
                          self.param.betasw[0],
                          self.param.betasw[1],
                          self.param.sigma2w_reg,
                          self.param.betastd ])
        
        opt = minimize(self.ll, beta0,  method='Nelder-Mead', options={'maxiter':5000, 'maxfev': 90000, 'ftol': 1e-1, 'disp': True})
        
        return opt
        

