
"""
Parameters class. Defines set of parameters
"""

class Parameters:
    """

	List of structural parameters and prices

	"""
    def __init__(self, betasw, betast, betasn, sigma2w, sigma2t, sigma2n, meanshocks, covshocks, T, Lc, alpha, gamma):
         
        self.betasw, self.betast, self.betasn = betasw, betast, betasn
        
        self.sigma2w, self.sigma2t, self.sigma2n = sigma2w, sigma2t, sigma2n
        
        self.T, self.Lc                 = T, Lc
        
        self.meanshocks, self.covshocks = meanshocks, covshocks
        
        self.alpha, self.gamma          = alpha, gamma
        
 