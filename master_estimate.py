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

#non-labor income
regn = sm.OLS(endog = data['ln_nli'], exog = data[['constant', 'married34', 'd_work', 'tot_kids', 'm_sch']], missing='drop').fit()

betasn  = regn.params
sigma2n = np.var(regn.resid)


#------------ PARAMETERS ------------#
betas_opt = np.load("/Users/jorge-home/Dropbox/Research/DN-early/Dynamic_childcare/Results/betas_modelv1.npy")

#the list of estimated parameters
alpha       = betas_opt[0]
gamma       = betas_opt[1]
meanshocks  = [betas_opt[2],betas_opt[3]]
sigma1      = 1#fixed
sigma2      = betas_opt[4]
rho         = betas_opt[5]
covshocks   = [sigma1,sigma2,rho]
betasw      = [betas_opt[6],betas_opt[7]]
sigma2w_reg = 0.34
betastd     = betas_opt[9]


T          = (24-8)*20  #monthly waking hours
Lc         = 8*20       #monthly cc hours
times = 50
times_boot = 1000


#------------ CALL CLASSES, ESTIMATION SIM & BOOTSTRAP ------------#
param0 = parameters.Parameters(betasw, betastd, betasn, sigma2, sigma2w_reg, meanshocks, covshocks, T, Lc, alpha, gamma, times)
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
    w_matrix[j,j] = (list_ses[j]**(-2))


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
betasw0_opt = opt.x[6]
betasw1_opt = opt.x[7]
sigma2w_opt = opt.x[8]
betastd_opt = opt.x[9]




betas_opt = np.array([alpha_opt,gamma_opt,meanshocks0_opt,meanshocks1_opt,
                      sigma2_shock_opt,rho_shock_opt,betasw0_opt,
                      betasw1_opt,sigma2w_opt,betastd_opt])


np.save('/Users/jorge-home/Dropbox/Research/DN-early/Dynamic_childcare/Results/betas_modelv1.npy',betas_opt)



















