#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 21:56:27 2020

@author: akac
"""
import numpy as np
import statsmodels.api as sm

class bootstrap:
    def __init__(self, N, data):
        "Initial class"
        
        self.N, self.data = N, data

            
    def boostr(self, times):
        n = self.data.shape[0]
        smpl_mean_labor = np.zeros(times)
        smpl_mean_cc    = np.zeros(times)
        smpl_mean_score = np.zeros(times)
        smpl_var_score  = np.zeros(times)
       # smpl_var_score_se = np.zeros(times)
        
        b0_est = np.zeros(times)
        b1_est = np.zeros(times)
        se0_est = np.zeros(times)
        se1_est = np.zeros(times)
        varres_est = np.zeros(times)
        
        b1_td_est = np.zeros(times)
        b1_tz_est = np.zeros(times)
        b1_dz_est = np.zeros(times)
        
        se_b1_td_est = np.zeros(times)
        se_b1_tz_est = np.zeros(times)
        se_b1_dz_est = np.zeros(times)
        
        
        for i in range(0,times):
            smpl = self.data.sample(n, replace=True)
            smpl_mean_labor[i] = np.mean(smpl['monthly_hrs']/160)
            smpl_mean_cc[i]    = np.mean(smpl['d_cc_34'])
            smpl_mean_score[i] = np.mean(smpl['TVIP_age_3'])
            smpl_var_score [i] = np.var(smpl['TVIP_age_3'])
            
            "log_w = beta_0 + beta_1*m_sch + e_i"
            reg_w=sm.OLS(endog=smpl['ln_w'], exog=smpl[['constant', 'm_sch']], missing='drop').fit()
            
            b0_est[i] = reg_w.params[0]
            b1_est[i] = reg_w.params[1]
            se0_est[i] = reg_w.bse[0] 
            se1_est[i] = reg_w.bse[1]
            varres_est[i] = np.var(reg_w.resid)     
            
            "Y_i = alpha_0 + alpha_1*D_i + u_i"
            reg_1  = sm.OLS(endog=smpl['TVIP_age_3'], exog=smpl[['constant', 'd_cc_34']], missing='drop').fit()
            b1_td_est[i]    = reg_1.params[1]
            se_b1_td_est[i] = reg_1.bse[1]
            
            "D_i = gamma_0 + gamma_1*Z_i + v_i"
            reg_2            = sm.OLS(endog=smpl['d_cc_34'], exog=smpl[['constant', 'commute2cc']], missing='drop').fit()
            b1_dz_est[i]    = reg_2.params[1]
            se_b1_dz_est[i] = reg_2.bse[1]
            
            "Y_i=beta_0+beta_1*Z_i + w_i"
            reg_3           = sm.OLS(endog=smpl['TVIP_age_3'], exog=smpl[['constant', 'commute2cc']], missing='drop').fit()
            b1_tz_est[i]    = reg_3.params[1] 
            se_b1_tz_est[i] = reg_3.bse[1]
            
        
        labor_boot = np.mean(smpl_mean_labor)
        cc_boot    = np.mean(smpl_mean_cc)
        score_boot = np.mean(smpl_mean_score)
        var_score_boot = np.mean(smpl_var_score)
        var_score_se = np.std(smpl_var_score)
        
        se_labor = np.std(smpl_mean_labor)
        se_cc    = np.std(smpl_mean_cc)
        se_score = np.std(smpl_mean_score)
        
        b0_boot = np.mean(b0_est)
        b1_boot = np.mean(b1_est)
        
        se0_boot  = np.mean(se0_est)
        se1_boot  = np.mean(se1_est)
        varres    = np.mean(varres_est) 
        varres_se = np.std(varres_est)

        
        b1_td_boot = np.mean(b1_td_est)
        b1_tz_boot = np.mean(b1_tz_est)
        b1_dz_boot = np.mean(b1_dz_est)
        se_b1_td_boot = np.std(b1_td_est)
        se_b1_tz_boot  = np.std(b1_tz_est)
        se_b1_dz_boot = np.std(b1_dz_est)
        


        return {'Labor Choice': labor_boot     , 'SE Labor Choice': se_labor  , 
                'CC Choice': cc_boot           , 'SE CC Choice': se_cc        , 
                'Test Score': score_boot       , 'SE Test Score': se_score    ,
                'beta0_w': b0_boot             , 'SE beta0_w': se0_boot         ,
                'beta1_w': b1_boot             , 'SE beta1_w': se1_boot         ,
                'resid_var_w': varres          , 'SE resid_var_w': varres_se    ,
                'resid_var_td': var_score_boot , 'SE resid_var_td': var_score_se ,
                'beta1_td': b1_td_boot         , 'SE beta1_td': se_b1_td_boot ,
                'beta1_tz': b1_tz_boot         , 'SE beta1_tz': se_b1_tz_boot   ,
                'beta1_dz': b1_dz_boot         , 'SE beta1_dz': se_b1_dz_boot }
    
    
    
    
    
    
    
    