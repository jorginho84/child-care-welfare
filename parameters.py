
"""
Parameters class. Defines set of parameters
"""

class Parameters:
    """

	List of structural parameters and prices

	"""
    def __init__(self, betas, betasw, betastd, betasn, sigma2n, sigma2w_estr, sigma2w_reg, meanshocks, covshocks, T, Lc, alpha, gamma, times):
         
        self.betas, self.betasw, self.betastd, self.betasn, self.sigma2n = betas, betasw, betastd, betasn, sigma2n
        
        self.sigma2w_estr, self.sigma2w_reg, =  sigma2w_estr, sigma2w_reg
        
        self.T, self.Lc  = T, Lc
        
        self.meanshocks, self.covshocks = meanshocks, covshocks
        
        self.alpha, self.gamma          = alpha, gamma
        
        self. times = times
        
