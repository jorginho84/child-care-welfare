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
data1 = pd.read_stata('data_python.dta')
N = len(data1)
data1['constant']=np.ones((N,1))
data=data1.to_numpy()

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

#------------ CALL FUNCTIONS ------------#
param0 = parameters.Parameters(betas,sigmaw, meanshocks, covshocks, T, Lc, alpha, gamma)

model = util.Utility(param0, N, data)

#de acá hacia abajo debería estar dentro de la función? 
wage   = model.waget()
shocks = model.res_causal()

U      = model.utility(shocks,wage, H1, D1)

#------------ DATA SIMULATION ------------#

sim = simdata.SimData(N, model, shocks, wage)
opt_set = sim.choice()


end_time = datetime.datetime.now()

print(end_time-begin_time)