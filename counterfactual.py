#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 22:31:08 2021

@author: akac
"""

import numpy as np
import pandas as pd
import pickle
import tracemalloc
import itertools
import sys, os
import time
from scipy import stats
from scipy import interpolate
import matplotlib.pyplot as plt
import statsmodels.api as sm
import xlsxwriter

#sys.path.append("/Users/jorge-home/Dropbox/Research/DN-early/structural")
sys.path.append("/Users/antoniaaguilera/Desktop/RA/python/sim-anto/currently working")


#call py scripts
import utility    as util
import parameters as parameters
import simdata    as simdata
import estimate  as est
import bootstrap  as bstr

np.random.seed(123)


#------------ PREP DATA ------------#

data = pd.read_stata('data/data_python.dta')
N = len(data)
data['constant'] = np.ones((N,1))

#------------ REG WITH DATA ------------#
#wage
regw=sm.OLS(endog=data['ln_w'], exog=data[['constant', 'm_sch']], missing='drop').fit()


#test score vs d_cc
regtd=sm.OLS(endog=data['TVIP_age_3'], exog=data[['constant', 'd_cc_34']], missing='drop').fit()

betastd = regtd.params

sigma2td= np.var(regtd.resid)

#d_cc vs commute
regdz=sm.OLS(endog=data['d_cc_34'], exog=data[['constant', 'commute2cc']], missing='drop').fit()

betasdz = regdz.params

#test score vs commute            
regtz = sm.OLS(endog=data['TVIP_age_3'], exog=data[['constant', 'commute2cc']], missing='drop').fit()

betastz = regtz.params
            

#non-labor income
regn = sm.OLS(endog = data['ln_nli'], exog = data[['constant', 'married34', 'd_work', 'tot_kids', 'm_sch']], missing='drop').fit()

betasn  = regn.params
sigma2n = np.var(regn.resid)


#------------ PARAMETERS ------------#
#betas_opt = np.load("/Users/jorge-home/Dropbox/Research/DN-early/Dynamic_childcare/Results/betas_modelv1.npy")
betas_opt = np.load("/Users/antoniaaguilera/Desktop/RA/python/sim-anto/currently working/betas_modelv2.npy")


#the list of estimated parameters
#alpha = betas_opt[0] #0.01-0.1 ver el fit en labor choice
alpha = 0.003
gamma = betas_opt[1]
meanshocks = [betas_opt[2],betas_opt[3]]
sigma1     = 1#fixed
sigma2     = betas_opt[4]
rho        = betas_opt[5]
covshocks  = [sigma1,sigma2,rho]
betasw = [betas_opt[6],betas_opt[7]]
sigma2w_reg = 0.34
betastd = betas_opt[9]


T          = (24-8)*20  #monthly waking hours
Lc         = 8*20       #monthly cc hours
w_matrix   = np.identity(10)
times = 1
times_boot = 1000 


#------------ SIMULATION A: BASELINE ------------#

param0 = parameters.Parameters(betasw, betastd, betasn, sigma2n, sigma2w_reg, meanshocks, covshocks, T, Lc, alpha, gamma, times)
model     = util.Utility(param0, N, data)
model_sim = simdata.SimData(N, model)
model_boot= bstr.bootstrap(N, data)

moments_boot = model_boot.boostr(times_boot)
model_est = est.estimate(N, data, param0, moments_boot, w_matrix)


results_estimate_a = model_est.simulation(model_sim)

a = (results_estimate_a['sim_Labor'], results_estimate_a['sim_CC'], results_estimate_a['sim_Score'])


# ------------ SIMULATION B: CHANGE DISTANCE ------------ #

delta_distance = 0.1
data['min_center_34_b'] = None
data['min_center_34_b'] = data['min_center_34']*(1-delta_distance)
data['commute2cc'] = (data['min_center_34_b']/14.5)*2*20  #monthly commute hours, 2 commutes a day 20 days a month 


param0 = parameters.Parameters(betasw, betastd, betasn, sigma2n, sigma2w_reg, meanshocks, covshocks, T, Lc, alpha, gamma, times)
model     = util.Utility(param0, N, data)
model_sim = simdata.SimData(N, model)
model_boot= bstr.bootstrap(N, data)

moments_boot = model_boot.boostr(times_boot)
model_est = est.estimate(N, data, param0, moments_boot, w_matrix)


results_estimate_b = model_est.simulation(model_sim)

b = (results_estimate_b['sim_Labor'], results_estimate_b['sim_CC'],results_estimate_b['sim_Score'])


# ------------ C: COMPLIERS ------------ #

a_aux = (a[1] == 0 ).all(axis=1) #True if sim_CC==0 in (a)
b_aux = (b[1] == 1 ).all(axis=1) #True if sim_CC==1 in (b)
c_aux = a_aux & b_aux            #True if sim_CC==0 in (a) & sim_CC==1 in (b)

print('CC=0 with original distance:', round(np.mean(a_aux)*100,2),'%')
print('CC=1 with',(1-delta_distance)*100,'% distance:', round(np.mean(b_aux)*100,2),'%')
print('Number of compliers:', sum(c_aux))

data['compliers'] = c_aux.astype(int)


# ------------ D: SOME CALCULATIONS ON COMPLIERS ------------ #

E_score = sum((b[2] - a[2])*np.reshape(np.array(data['compliers']), (N,1)))/sum(data['compliers'])
E_work  = sum((b[0] - a[0])*np.reshape(np.array(data['compliers']), (N,1)))/sum(data['compliers'])
print('E(score_b-score_a)=', E_score)
print('E(work_b-work_a)=', E_work)



