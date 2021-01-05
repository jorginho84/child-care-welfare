#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 21:56:27 2020

@author: akac
"""
import numpy as np
import statsmodels.api as sm


def bootstrap(data, times):
        n = data.shape[0]
        smpl_mean_labor = np.zeros(times)
        smpl_mean_cc    = np.zeros(times)
        smpl_mean_score = np.zeros(times)
        smpl_var_score  = np.zeros(times)
        smpl_var_score_se = np.zeros(times)
        
        b0_est = np.zeros(times)
        b1_est = np.zeros(times)
        se0_est = np.zeros(times)
        se1_est = np.zeros(times)
        varres_est = np.zeros(times)
        
        alpha1_est = np.zeros(times)
        beta1_est  = np.zeros(times)
        gamma1_est = np.zeros(times)
        
        se_alpha1_est = np.zeros(times)
        se_beta1_est  = np.zeros(times)
        se_gamma1_est = np.zeros(times)
        
        
        for i in range(0,times):
            smpl = data.sample(n, replace=True)
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
            alpha1_est[i]    = reg_1.params[1]
            se_alpha1_est[i] = reg_1.bse[1]
            
            "D_i = gamma_0 + gamma_1*Z_i + v_i"
            reg_2            = sm.OLS(endog=smpl['d_cc_34'], exog=smpl[['constant', 'commute2cc']], missing='drop').fit()
            gamma1_est[i]    = reg_2.params[1]
            se_gamma1_est[i] = reg_2.bse[1]
            
            "Y_i=beta_0+beta_1*Z_i + w_i"
            reg_3           = sm.OLS(endog=smpl['TVIP_age_3'], exog=smpl[['constant', 'commute2cc']], missing='drop').fit()
            beta1_est[i]    = reg_3.params[1] 
            se_beta1_est[i] = reg_3.bse[1]
            
        
        labor_boot = np.mean(smpl_mean_labor)
        cc_boot    = np.mean(smpl_mean_cc)
        score_boot = np.mean(smpl_mean_score)
        var_score_boot = np.mean(smpl_var_score)
        var_score_se = np.std(smpl_var_score)/np.sqrt(n)
        
        se_labor = np.std(smpl_mean_labor)/np.sqrt(n)
        se_cc    = np.std(smpl_mean_cc)/np.sqrt(n)
        se_score = np.std(smpl_mean_score)/np.sqrt(n)
        
        b0_boot = np.mean(b0_est)
        b1_boot = np.mean(b1_est)
        
        se0_boot  = np.mean(se0_est)
        se1_boot  = np.mean(se1_est)
        varres    = np.mean(varres_est) 
        varres_se = np.std(varres_est)/np.sqrt(n)

        
        alpha1_boot = np.mean(alpha1_est)
        beta1_boot  = np.mean(beta1_est)
        gamma1_boot = np.mean(gamma1_est)
        se_alpha1_boot = np.mean(se_alpha1_est)
        se_beta1_boot  = np.mean(se_beta1_est)
        se_gamma1_boot = np.mean(se_gamma1_est)
        


        return {'Labor Choice': labor_boot     , 'SE Labor Choice': se_labor  , 
                'CC Choice': cc_boot           , 'SE CC Choice': se_cc        ,
                'Test Score': score_boot       , 'SE Test Score': se_score    ,
                'Beta0': b0_boot               , 'SE Beta0': se0_boot         ,
                'Beta1': b1_boot               , 'SE Beta1': se1_boot         ,
                'Resid var': varres            , 'SE sigma^2_e': varres_se    ,
                'Var Score': var_score_boot    , 'SE Var Score': var_score_se ,
                'alpha_1': alpha1_boot         , 'SE alpha_1': se_alpha1_boot ,
                'beta_1' : beta1_boot          , 'SE beta_1': se_beta1_boot   ,
                'gamma_1': gamma1_boot         , 'SE gamma_1': se_gamma1_boot }
    
    
    
    
    
    
    
    
    
    
    