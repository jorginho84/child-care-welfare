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
#alpha = betas_opt[0]
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
times = 50
times_boot = 1000 



#------------ CALL CLASSES, ESTIMATION SIM & BOOTSTRAP ------------#
param0 = parameters.Parameters(betasw, betastd, betasn, sigma2n, sigma2w_reg, meanshocks, covshocks, T, Lc, alpha, gamma, times)
model     = util.Utility(param0, N, data)
model_sim = simdata.SimData(N, model)
model_boot= bstr.bootstrap(N, data)

moments_boot = model_boot.boostr(times_boot)
model_est = est.estimate(N, data, param0, moments_boot, w_matrix)


results_estimate = model_est.simulation(model_sim)


#------------ EXCEL TABLE ------------#

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
worksheet.write('E6', moments_boot['SE beta0_w'])
worksheet.write('E7', moments_boot['SE beta1_w'])
worksheet.write('E8', moments_boot['SE resid_var_w'])
worksheet.write('E9', moments_boot['SE beta1_td'])
worksheet.write('E10', moments_boot['SE beta1_tz'])
worksheet.write('E11', moments_boot['SE beta1_dz'])
worksheet.write('E12', moments_boot['SE resid_var_td'])


workbook.close()

# ------------ WRITE .TEX FILE ------------ # 

#with open("/Users/jorge-home/Dropbox/Research/DN-early/structural/tabla1.tex", "w")
with open("/Users/antoniaaguilera/Desktop/RA/python/sim-anto/currently working/tabla1.tex", "w") as f:   # Opens file and casts as f 
    f.write("\\begin{tabular}{lcccccc}   \n")
    f.write("\\toprule\\textbf{Moments} & & \\textbf{Simulated} & & \\textbf{Data} & & \\textbf{S.E. data} \\\\\hline \n")
    f.write("A. Labor supply, child care choice and test score &&         & &       & &  \\\\ \n")
    f.write("Pr(full-time work)  & &" + str(round(results_estimate['Labor Choice'], 3)) + "& &" + str(round(moments_boot['Labor Choice'],3)) + "& &" + str(round(moments_boot['SE Labor Choice'],3)) + " \\\\ \n")
    f.write("Pr(cc choice)       & &" + str(round(results_estimate['CC Choice'],3))     + "& &" + str(round(moments_boot['CC Choice'],3))    + "& &" + str(round(moments_boot['SE CC Choice'],3))    + " \\\\ \n")
    f.write("Test Score          & &" + str(round(results_estimate['Test Score'],3))    + "& &" + str(round(moments_boot['Test Score'],3))   + "& &" + str(round(moments_boot['SE Test Score'],3))   + " \\\\ \n")
    f.write(" & & & & & & \\\\ \n ")
    f.write("B. Mother: log(wage_i)=X'_i\\beta +\\varepsilon_i & & & & && \\\\ \n")
    f.write("Constant            & &" + str(round(results_estimate['beta0_w'],3))     + "& &" + str(round(moments_boot['beta0_w'],3))     + "& &" + str(round(moments_boot['SE beta0_w'],3))     + " \\\\ \n")
    f.write("Return to schooling & &" + str(round(results_estimate['beta1_w'],3))     + "& &" + str(round(moments_boot['beta1_w'],3))     + "& &" + str(round(moments_boot['SE beta1_w'],3))     + " \\\\ \n")
    f.write("\\sigma^2           & &" + str(round(results_estimate['resid_var_w'],3)) + "& &" + str(round(moments_boot['resid_var_w'],3)) + "& &" + str(round(moments_boot['SE resid_var_w'],3)) + " \\\\ \n")
    f.write(" & & & & & & \\\\ \n ")
    f.write("C. Child: score_i = X'_i\\beta + \\varepsilon_i   & &  & &  & &  \\\\ \n")
    f.write("Coefficient on child care dummy & &" + str(round(results_estimate['beta1_td'],3))+ "& &" + str(round(moments_boot['beta1_td'],3))     + "& &" + str(round(moments_boot['SE beta1_td'],3))     + " \\\\ \n")
    f.write("\\sigma^2                  & &" + str(round(results_estimate['resid_var_td'],3)) + "& &" + str(round(moments_boot['resid_var_td'],3)) + "& &" + str(round(moments_boot['SE resid_var_td'],3)) + " \\\\ \n")
    f.write(" & & & & & & \\\\ \n ")
    f.write("D. Child: score_i = Z'_i\\delta + u_i   & &  & &  & &  \\\\ \n")
    f.write("Coefficient on commute time to child care center  & &" + str(round(results_estimate['beta1_tz'],3))+ "& &" + str(round(moments_boot['beta1_tz'],3)) + "& &" + str(round(moments_boot['SE beta1_tz'],3)) + " \\\\ \n")
    f.write(" & & & & & & \\\\ \n ")
    f.write("E. Child: d\\_cc_i = X'_i\\beta + \\varepsilon_i   & &  & &  & &  \\\\ \n")
    f.write("Coefficient on commute time to child care center & &" + str(round(results_estimate['beta1_dz'],3))+ "& &" + str(round(moments_boot['beta1_dz'],3)) + "& &" + str(round(moments_boot['SE beta1_dz'],3)) + " \\\\ \n") 
    f.write("\end{tabular}")
















