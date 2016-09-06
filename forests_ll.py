from __future__ import division
import numpy as np
import csv
from scipy.special import expit, logit
from scipy.misc import factorial
from scipy.optimize import bisect

# Logseries solver from macroecotools/macroeco_distributions
def logser_solver(S, N):
    """Given abundance data, solve for MLE of logseries parameter p."""
    BOUNDS = [0, 1]
    DIST_FROM_BOUND = 10 ** -15    
    y = lambda x: 1 / np.log(1 / (1 - expit(x))) * expit(x) / (1 - expit(x)) - N / S
    x = bisect(y, logit(BOUNDS[0] + DIST_FROM_BOUND), logit(BOUNDS[1] - DIST_FROM_BOUND),
                xtol = 1.490116e-08)
    nu = 1 - expit(x)
    return nu

def import_raw_data(input_filename):
    data = np.genfromtxt(input_filename, dtype = "S15, S25, f8", skip_header = 1, 
                         names = ['site', 'sp', 'dbh'], delimiter = ",")
    return data

def get_model_lik(comm_dat, min_S = 9):
    """Compute the likelihood for the 4 models (2 neutral, 2 maxent) 
    
    given a community array with 3 columns: 'site', 'sp', and 'dbh'.
    Returns a list with the four likelihood values.
    
    """
    sp_list = np.unique(comm_dat['sp'])
    if len(sp_list) > min_S:
        S = len(np.unique(comm_dat['sp']))
        N = len(comm_dat)
        M = sum(comm_dat['dbh']) # For now M is defind for dbh 
        
        # Parameter estimation
        nu = logser_solver(S, N)
        m0 = M / N
        lambda1 = np.log(N / (N - S))
        lambda2 = S / M
        
        # log-likelihoods
        ssnt, ssnti, ssme, ssmei = 0, 0, 0, 0
        for sp in sp_list:
            m_sp = comm_dat['dbh'][comm_dat['sp'] == sp]
            n = len(m_sp)
            M_sp = sum(m_sp)
            ssnt += -np.log(m0) - np.log(-np.log(nu)) + n * np.log(1 - nu) - sum(np.log(range(1, n + 1))) + \
                (n - 1) * np.log(M_sp / m0) - M_sp / m0
            ssnti += -np.log(-np.log(nu)) + n * np.log(1 - nu) - np.log(n) - n * np.log(m0) - M_sp / m0
            ssme += np.log(N / (N - S) - 1) + np.log(lambda2) - n * lambda1 - M_sp * lambda2
            ssmei += np.log(N / (N - S) - 1) + n * np.log(lambda2) - n * lambda1 - lambda2 * M_sp
            
        return [ssnt, ssnti, ssme, ssmei]
    
dat_dir = 'C:\\Users\\Xiao\\Dropbox\\projects\\mete_neutral_comp\\data\\'
dat_list = ['ACA', 'BCI', 'BVSF', 'CSIRO', 'FERP', 'Lahei', 'LaSelva', 'NC', 'Oosting', 'Serimbu', 
            'WesternGhats', 'Cocoli', 'Luquillo', 'Sherman', 'Shirakami']
out_dir = 'C:\\Users\\Xiao\\Dropbox\\projects\\maxent_null\\4model_ll.csv'
writer_link = open(out_dir, 'wb')
writer = csv.writer(writer_link)
writer.writerow(['data', 'site', 'SSNT', 'SSNTI', 'SSME', 'SSMEI'])

for dat_name in dat_list:
    dat = import_raw_data(dat_dir + dat_name + '.csv')
    for site in np.unique(dat['site']):
        dat_site = dat[dat['site'] == site]
        ll_site = get_model_lik(dat_site)
        if ll_site is not None:
            writer.writerow([dat_name, site] + ll_site)

writer_link.close()
