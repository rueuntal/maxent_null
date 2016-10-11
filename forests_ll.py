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

def log_pssnt(m0, nu, n, M_sp):
    log_p = -np.log(m0) - np.log(np.log(1 / nu)) + n * np.log(1 - nu) - \
        sum(np.log(range(1, n + 1))) + (n - 1) * np.log(M_sp / m0) - M_sp / m0
    return log_p

def log_pssme(lambda1, lambda2, n, M_sp):
    log_p = np.log(np.exp(lambda1) - 1) + np.log(lambda2) - n * lambda1 - M_sp * lambda2
    return log_p

def  log_pssnti(m0, nu, n, m_list):
    log_p = -np.log(np.log(1 / nu)) + n * np.log(1 - nu) - np.log(n) - n * np.log(m0) + \
        sum(-m_list / m0)
    return log_p

def log_pssmei(lambda1, lambda2, n, m_list):
    log_p = np.log(np.exp(lambda1) - 1) + n * np.log(lambda2) - n * lambda1 + \
        sum(-m_list * lambda2)
    return log_p

def get_model_lik(comm_dat, min_S = 9):
    """Compute the likelihood for the 4 models (2 neutral, 2 maxent) 
    
    given a community array with 3 columns: 'site', 'sp', and 'dbh'.
    Returns a list with the four likelihood values.
    
    """
    sp_list = np.unique(comm_dat['sp'])
    if len(sp_list) > min_S:
        comm_mass = comm_dat['dbh'] ** 8/3 # convert dbh to mass
        S = len(np.unique(comm_dat['sp']))
        N = len(comm_dat)
        M = sum(comm_mass) 
        
        # Parameter estimation
        nu = logser_solver(S, N)
        m0 = M / N
        lambda1 = np.log(N / (N - S))
        lambda2_ssme = S / M
        lambda2_ssmei = N / M
        
        # log-likelihoods
        ssnt, ssnti, ssme, ssmei = 0, 0, 0, 0
        for sp in sp_list:
            m_sp = comm_mass[comm_dat['sp'] == sp]
            n = len(m_sp)
            M_sp = sum(m_sp)
            ssnt += log_pssnt(m0, nu, n, M_sp)
            ssnti += log_pssnti(m0, nu, n, m_sp)
            ssme += log_pssme(lambda1, lambda2_ssme, n, M_sp)
            ssmei += log_pssmei(lambda1, lambda2_ssmei, n, m_sp)
            
        return [ssnt, ssnti, ssme, ssmei]
    
dat_dir = 'C:\\Users\\Xiao\\Dropbox\\projects\\mete_neutral_comp\\data\\'
dat_list = ['ACA', 'BCI', 'BVSF', 'CSIRO', 'FERP', 'LaSelva', 'NC', 'Oosting', 'Serimbu', 
            'WesternGhats', 'Cocoli', 'Luquillo', 'Sherman']
out_dir = 'C:\\Users\\Xiao\\Dropbox\\projects\\maxent_null\\4model_ll_updated.csv'
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

# Examine P_M_SSNT & P_M_EXP in BCI
out_dir_bci = 'C:\\Users\\Xiao\\Dropbox\\projects\\maxent_null\\bci_ll_comp.csv'
writer_link = open(out_dir_bci, 'wb')
writer = csv.writer(writer_link)
writer.writerow(['sp', 'n', 'M', 'P_SSNT', 'P_EXP'])

dat_bci = import_raw_data(dat_dir + 'BCI.csv')
S = len(np.unique(dat_bci['sp']))
N = len(dat_bci)
Mtot = sum(dat_bci['dbh'])
m0 = Mtot / N
lambda2 = S / Mtot
for sp in np.unique(dat_bci['sp']):
    dat_sp = dat_bci[dat_bci['sp'] == sp]
    M = sum(dat_sp['dbh'])
    n = len(dat_sp)
    ll_ssnt = -np.log(m0) - sum(np.log(range(1, n))) + (n - 1)  * np.log(M / m0) - M / m0
    ll_exp = np.log(lambda2) - lambda2 * M
    writer.writerow([sp, n, M, ll_ssnt, ll_exp])

writer_link.close()

# This is a sanity check
# The difference between SSNTI and SSMEI should just be the difference from SAD
def diff_sad(comm_dat):
    S = len(np.unique(comm_dat['sp']))
    N = len(comm_dat)
    nu = logser_solver(S, N)
    lambda1 = np.log(N / (N - S))
    lsad_ssnti, lsad_ssmei = 0, 0
    for sp in np.unique(comm_dat['sp']):
        n = len(comm_dat[comm_dat['sp'] == sp])
        lsad_ssnti += -np.log(n) - np.log(np.log(1 / nu)) + n * np.log(1 - nu)
        lsad_ssmei += np.log(np.exp(lambda1) - 1) - n * lambda1
    return lsad_ssnti - lsad_ssmei

for dat_name in dat_list:
    dat = import_raw_data(dat_dir + dat_name + '.csv')
    for site in np.unique(dat['site']):
        site_dat = dat[dat['site'] == site]
        print dat_name, site, diff_sad(site_dat)
        