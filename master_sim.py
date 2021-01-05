import numpy as np
import pandas as pd
import pickle
import tracemalloc
import itertools
import sys, os
import datetime
from scipy import stats
from scipy import interpolate
import matplotlib.pyplot as plt

sys.path.append("")

#call py scripts
import utility_akac as util
import parameters_akac as parameters
import simdata_akac as simdata

np.random.seed(123)

begin_time = datetime.datetime.now()
#------------ PREP DATA ------------#
<<<<<<< HEAD
data = pd.read_stata('/Users/antoniaaguilera/Documents/GitHub/child-care-welfare/data/data_python.dta')
N = len(data)
data['constant']=np.ones((N,1))

#------------ REG WITH DATA ------------#
#wage
regw=sm.OLS(endog=data['ln_w'], exog=data[['constant', 'm_sch']], missing='drop').fit()

#betas=[beta0, beta1]
betasw = regw.params
sigma2w= np.var(regw.resid)

#test score
regt=sm.OLS(endog=data['TVIP_age_3'], exog=data[['constant', 'd_cc_34']], missing='drop').fit()

betast = regt.params
sigma2t= np.var(regt.resid)

#non-labor income
regn = sm.OLS(endog = data['ln_nli'], exog = data[['constant', 'married34', 'd_work', 'tot_kids', 'm_sch']], missing='drop').fit()

betasn  = regn.params
sigma2n = np.var(regn.resid)
=======
data1 = pd.read_stata('data_python.dta')
N = len(data1)
data1['constant']=np.ones((N,1))
data=data1.to_numpy()
>>>>>>> parent of 56b0066... Semana 15-Dic

#------------ PARAMETERS ------------#
#betas  = [beta1 , beta0]
betas      = [0.0992312, 0.0084627] 
sigmaw     = 0.5869
meanshocks = [0,0]
covshocks  = [[0.5,0],[0,0.5]] 
T          = (24-8)*20  #monthly waking hours
Lc         = 8*20       #monthly cc hours
alpha      = 0.1
gamma      = 0.1

#------------ VARIABLES ------------#
H1 = np.reshape(data[:,3], (N,1))
D1 = np.reshape(data[:,2], (N,1))

<<<<<<< HEAD
H1 = np.array(data[['monthly_hrs']], dtype=np.float64)
D1 = np.array(data[['d_cc_34']], dtype=np.float64)

#------------ CALL CLASSES ------------#
param0 = parameters.Parameters(betasw, betast, betasn, sigma2w, sigma2t, sigma2n, 
                               meanshocks, covshocks, T, Lc, alpha, gamma)
    
model     = util.Utility(N, data, param0)
model_sim = simdata.SimData(N, model)
model_est = est.estimate(N, data, param0, model, model_sim)


#------------ ESTIMATION SIM & BOOTSTRAP ------------#
times = 20
results_estimate = model_est.simulation(times)
results_bootstrap = bstr.bootstrap(data, times)
=======
#------------ CALL FUNCTIONS ------------#
param0 = parameters.Parameters(betas,sigmaw, meanshocks, covshocks, T, Lc, alpha, gamma)

model = util.Utility(param0, N, data)

#de acá hacia abajo debería estar dentro de la función? 
wage   = model.waget()
shocks = model.res_causal()
>>>>>>> parent of 56b0066... Semana 15-Dic

U      = model.utility(shocks,wage, H1, D1)

#------------ DATA SIMULATION ------------#

<<<<<<< HEAD
worksheet.write('B2', 'parameter')
worksheet.write('B3', 'labor choice')
worksheet.write('B4', 'cc choice')
worksheet.write('B5', 'test score')
worksheet.write('B6', 'wage ec: beta_0')
worksheet.write('B7', 'wage ec: beta_1')
worksheet.write('B8', 'sigma^2_{varepsilon}')
worksheet.write('B9', 'alpha_1')
worksheet.write('B10', 'beta_1')
worksheet.write('B11', 'gamma_1')
worksheet.write('B12', 'var score')


worksheet.write('C2', 'sim')
worksheet.write('C3', results_estimate['Labor Choice'])
worksheet.write('C4', results_estimate['CC Choice'])
worksheet.write('C5', results_estimate['Test Score'])
worksheet.write('C6', results_estimate['Beta0'])
worksheet.write('C7', results_estimate['Beta1'])
worksheet.write('C8', results_estimate['Resid var'])
worksheet.write('C9', results_estimate['alpha_1'])
worksheet.write('C10', results_estimate['beta_1'])
worksheet.write('C11', results_estimate['gamma_1'])
worksheet.write('C12', results_estimate['Var Score'])


worksheet.write('D2', 'data')
worksheet.write('D3', results_bootstrap['Labor Choice'])
worksheet.write('D4', results_bootstrap['CC Choice'])
worksheet.write('D5', results_bootstrap['Test Score'])
worksheet.write('D6', results_bootstrap['Beta0'])
worksheet.write('D7', results_bootstrap['Beta1'])
worksheet.write('D8', results_bootstrap['Resid var'])
worksheet.write('D9', results_bootstrap['alpha_1'])
worksheet.write('D10', results_bootstrap['beta_1'])
worksheet.write('D11', results_bootstrap['gamma_1'])
worksheet.write('D12', results_bootstrap['Var Score'])


worksheet.write('E2', 'SE')
worksheet.write('E3', results_bootstrap['SE Labor Choice'])
worksheet.write('E4', results_bootstrap['SE CC Choice'])
worksheet.write('E5', results_bootstrap['SE Test Score'])
worksheet.write('E6', results_bootstrap['SE Beta0'])
worksheet.write('E7', results_bootstrap['SE Beta1'])
worksheet.write('E8', results_bootstrap['SE sigma^2_e'])
worksheet.write('E9', results_bootstrap['SE alpha_1'])
worksheet.write('E10', results_bootstrap['SE beta_1'])
worksheet.write('E11', results_bootstrap['SE gamma_1'])
worksheet.write('E12', results_bootstrap['SE Var Score'])



        
workbook.close()
#------------ END TIME ------------#
=======
sim = simdata.SimData(N, model, shocks, wage)
opt_set = sim.choice()


>>>>>>> parent of 56b0066... Semana 15-Dic
end_time = datetime.datetime.now()

print(end_time-begin_time)