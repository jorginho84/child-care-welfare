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
betasw = [-0.39,0.15]
sigma2w_reg = np.var(regw.resid)

#test score vs d_cc
regtd=sm.OLS(endog=data['TVIP_age_3'], exog=data[['constant', 'd_cc_34']], missing='drop').fit()

betastd = regtd.params
betastd = [-0.05]
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
meanshocks = [-0.1,0.5]
rho        = 0.75
sigma1     = 1#constante
sigma2     = 0.9
covshocks  = [sigma1,sigma2,rho]
T          = (24-8)*20  #monthly waking hours
Lc         = 8*20       #monthly cc hours
alpha      = -0.5
gamma      = 0.4

times = 50
times_boot = 1000






#------------ CALL CLASSES, ESTIMATION SIM & BOOTSTRAP ------------#
param0 = parameters.Parameters(betas, betasw, betastd, betasn, sigma2n, sigma2w_estr, sigma2w_reg, meanshocks, covshocks, T, Lc, alpha, gamma, times)
model     = util.Utility(param0, N, data)
model_sim = simdata.SimData(N, model)
model_boot= bstr.bootstrap(N, data)

moments_boot = model_boot.boostr(times_boot)

w_matrix = np.zeros((10,10))

list_ses = [moments_boot['SE Labor Choice'],
            moments_boot['SE CC Choice'],
            moments_boot['SE Test Score'],
            moments_boot['SE Beta0'],
            moments_boot['SE Beta1'],
            moments_boot['SE sigma^2_e'],
            moments_boot['SE beta1_td'],
            moments_boot['SE beta1_tz'],
            moments_boot['SE beta1_dz'],
            moments_boot['SE Var Score']]

for j in range(10):
    w_matrix[j,j] = (list_ses[j]**2)**(-1)


model_est = est.estimate(N, data, param0, moments_boot, w_matrix)




#----------------- OPTIMIZER -----------------#

start_time = time.time()

opt = model_est.optimizer()

#------------ END TIME ------------#


time_opt=time.time() - start_time
print ('Done in')
print("--- %s seconds ---" % (time_opt))

        

#the list of estimated parameters
alpha_opt = opt.x[0]
gamma_opt = opt.x[1]
meanshocks0_opt = opt.x[2]
meanshocks1_opt = opt.x[3]
sigma2_shock_opt = opt.x[4]
rho_shock_opt = opt.x[5]
betas0_opt = opt.x[6]
betas1_opt = opt.x[7]
sigma2w_opt = opt.x[8]
betastd_opt = opt.x[9]




betas_opt = np.array([alpha_opt,gamma_opt,meanshocks0_opt,meanshocks1_opt,
                      sigma2_shock_opt,rho_shock_opt,betas0_opt,
                      betas1_opt,sigma2w_opt,betastd_opt])


np.save('/Users/jorge-home/Dropbox/Research/DN-early/Dynamic_childcare/Results/betas_modelv1.npy',betas_opt)



















