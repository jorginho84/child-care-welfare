
"""
Parameters class. Defines set of parameters
"""

class Parameters:
    """

	List of structural parameters and prices

	"""
    def __init__(self, betas, sigmaw, meanshocks, covshocks, T, Lc, alpha, gamma):
        
        self.betas, self.sigmaw,        = betas, sigmaw
        
        self.T, self.Lc                 = T, Lc
        
        self.meanshocks, self.covshocks = meanshocks, covshocks
        
        self.alpha, self.gamma          = alpha, gamma
    