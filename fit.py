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

sys.path.append("/Users/jorge-home/Dropbox/Research/DN-early/structural")
#sys.path.append("/Users/antoniaaguilera/Desktop/AY-INV/python/sim-anto/currently working")


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

betasw = regw.params
sigma2w_reg = np.var(regw.resid)

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
#betas  = [beta1 , beta0]
betas      = [0.0992312, 0.0084627] 
sigma2w_estr = 0.5869
meanshocks = [0,0.5]
rho        = 0.9
sigma1     = 1#constante
sigma2     = 0.9
covshocks  = [sigma1,sigma2,rho]
T          = (24-8)*20  #monthly waking hours
Lc         = 8*20       #monthly cc hours
alpha      = -0.1
gamma      = 0.4
w_matrix   = np.identity(10)
times = 20



#------------ CALL CLASSES, ESTIMATION SIM & BOOTSTRAP ------------#
param0 = parameters.Parameters(betas, betasw, betastd, betasn, sigma2w_estr, sigma2w_reg, meanshocks, covshocks, T, Lc, alpha, gamma, times)
model     = util.Utility(param0, N, data)
model_sim = simdata.SimData(N, model)
model_boot= bstr.bootstrap(N, data)

moments_boot = model_boot.boostr(times)
model_est = est.estimate(N, data, param0, moments_boot, w_matrix)


results_estimate = model_est.simulation(model_sim)


#------------ DATA SIMULATION ------------#

workbook = xlsxwriter.Workbook('data/labor_choice.xlsx')
worksheet = workbook.add_worksheet()

worksheet.write('B2', 'parameter')
worksheet.write('B3', 'labor choice')
worksheet.write('B4', 'cc choice')
worksheet.write('B5', 'test score')
worksheet.write('B6', 'wage ec: beta_0')
worksheet.write('B7', 'wage ec: beta_1')
worksheet.write('B8', 'sigma^2_{varepsilon}')
worksheet.write('B9', 'beta1_td')
worksheet.write('B10', 'beta1_tz')
worksheet.write('B11', 'beta1_dz')
worksheet.write('B12', 'resid var score')


worksheet.write('C2', 'sim')
worksheet.write('C3', results_estimate['Labor Choice'])
worksheet.write('C4', results_estimate['CC Choice'])
worksheet.write('C5', results_estimate['Test Score'])
worksheet.write('C6', results_estimate['beta0_w'])
worksheet.write('C7', results_estimate['beta1_w'])
worksheet.write('C8', results_estimate['resid_var_w'])
worksheet.write('C9', results_estimate['beta1_td'])
worksheet.write('C10', results_estimate['beta1_tz'])
worksheet.write('C11', results_estimate['beta1_dz'])
worksheet.write('C12', results_estimate['resid_var_td'])
        
        
worksheet.write('D2', 'data')
worksheet.write('D3', moments_boot['Labor Choice'])
worksheet.write('D4', moments_boot['CC Choice'])
worksheet.write('D5', moments_boot['Test Score'])
worksheet.write('D6', moments_boot['beta0_w'])
worksheet.write('D7', moments_boot['beta1_w'])
worksheet.write('D8', moments_boot['resid_var_w'])
worksheet.write('D9', moments_boot['beta1_td'])
worksheet.write('D10', moments_boot['beta1_tz'])
worksheet.write('D11', moments_boot['beta1_dz'])
worksheet.write('D12', moments_boot['resid_var_td'])


worksheet.write('E2', 'SE')
worksheet.write('E3', moments_boot['SE Labor Choice'])
worksheet.write('E4', moments_boot['SE CC Choice'])
worksheet.write('E5', moments_boot['SE Test Score'])
worksheet.write('E6', moments_boot['SE Beta0'])
worksheet.write('E7', moments_boot['SE Beta1'])
worksheet.write('E8', moments_boot['SE sigma^2_e'])
worksheet.write('E9', moments_boot['SE beta1_td'])
worksheet.write('E10', moments_boot['SE beta1_tz'])
worksheet.write('E11', moments_boot['SE beta1_dz'])
worksheet.write('E12', moments_boot['SE Var Score'])


workbook.close()






















