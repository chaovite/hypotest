"""Utility functions for hypothesis tests and visualization"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd

def rANOVA(*args):
    """return the F statistics and p value after applying r-ANOVA
    args: array like, sample1, sample 2, ...
    ref: http://vassarstats.net/textbook/ch12a.html
    """
    # convert *args into dataframe.
    data = np.vstack(args).T
    df   = pd.DataFrame(data, dtype=float)
    n,   k  = df.shape
    n_T     = n*k
    M_k     = df.mean()
    M       = M_k.mean()
    SS_bg   = (df.sum()**2).sum()/n - (df.sum().sum())**2/n_T
    SS_wg   = ((df**2).sum() - df.sum()**2/n).sum()
    SS_subj = (df.sum(axis=1)**2).sum()/k - (df.sum().sum())**2/n_T
    SS_err  = SS_wg - SS_subj
    df_T    = n_T - 1
    df_bg   = k - 1
    df_wg   = n_T - k
    df_subj = n - 1
    df_err  = df_wg - df_subj
    MS_bg   = SS_bg/df_bg
    MS_err  = SS_err/df_err
    F       = MS_bg/MS_err
    pvalue  = 1 - scipy.stats.f.cdf(F, df_bg, df_err) 
    return F, pvalue

def plot_dignostics(samples, labels = None, test = None , bins=15):
    """
    make box plots for multiple samples and perform statistical tests.
    """
    n=len(samples)
    # box plots.
    if not labels:
        labels = ['X%d'%(i) for i in range(n)] 
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.boxplot(samples, positions=list(range(n)), labels=labels, widths=0.5)
    
    plt.subplot(1,2,2)
    for i in range(n):
        plt.hist(samples[i],alpha=0.8,bins=bins)
    plt.legend(labels)
    
    if not test:
        return
    test = test.lower()
    implemented_tests = ['kruskal','f','friedman','ranova']
    
    if test=='f':
        T, pvalue = scipy.stats.f_oneway(*samples)
    elif test=='kruskal':
        T, pvalue = scipy.stats.kruskal(*samples)
    elif test=='friedman':
        T, pvalue = scipy.stats.friedmanchisquare(*samples)
    elif test=='ranova':
        T, pvalue = rANOVA(*samples)
    else:
        print('%s is not supported，please use: %s'%(test,','.join(implemented_tests)))
        return None
    
    if test in implemented_tests:
        print('%s test statistics T: %8.5f and P-vale %10.8f' % (test,T, pvalue))
        if pvalue<0.05:
            print('Reject null hypothesis at 95% confidence')
        else:
            print('Fail to reject null hypothesis at 95% confidence')
        return T, pvalue

def Ftest_eq_vars(*args):
    """two tail F test of equal variance"""
    X, Y = args
    n, m = len(X), len(Y)
    dfn, dfd = n-1, m-1
    Sx = X.var(ddof=1)
    Sy = Y.var(ddof=1)
    T  = Sx/Sy
    lo, hi = min(T, 1/T), max(T,1/T)
    pvalue = 1 - scipy.stats.f.cdf(hi, dfn, dfd) + scipy.stats.f.cdf(lo, dfn, dfd)
    return T, pvalue
    
def homo_vars(samples, plot=False, labels = None, bins=15, test = None, **kwds):
    """
    test the homogeneity of variance
    test : {'f','levene','bartlett'}, 
    when test takes 'levene', the following keywords may be added:
        center : {'mean', 'median', 'trimmed'}, optional
            Which function of the data to use in the test.  The default
            is 'median'.
        proportiontocut : float, optional
            When `center` is 'trimmed', this gives the proportion of data points
            to cut from each end. (See `scipy.stats.trim_mean`.)
            Default is 0.05.
    """
    
    # make box plots
    n=len(samples)
    # box plots.
    if not labels:
        labels = ['X%d'%(i) for i in range(n)] 
    if plot:
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.boxplot(samples, positions=list(range(n)), labels=labels, widths=0.5)
        plt.subplot(1,2,2)
        for i in range(n):
            plt.hist(samples[i],alpha=0.8,bins=bins)
        plt.legend(labels)
    
    if not test:
        return
    tests_implemented = ('f','levene','bartlett')
    if not test in tests_implemented:
        raise ValueError("Keyword argument <test> must be 'f', 'levene'"
              + "or 'bartlett'.")
    # default center and proportionaltocut for levene test
    center = 'median'
    proportiontocut = 0.05
    
    for kw, value in kwds.items():
        if kw not in ['center', 'proportiontocut']:
            raise TypeError("levene() got an unexpected keyword argument '%s'" % kw)
        if kw == 'center':
            center = value
        else:
            proportiontocut = value
    if not center in ['mean','median','trimmed']:
        raise ValueError("Keyword argument <center> must be 'mean', 'median'"
              + "or 'trimmed'.")
    
    if test=='f':
        T, pvalue = Ftest_eq_vars(*samples)
    elif test=='bartlett':
        T, pvalue = scipy.stats.bartlett(*samples)
    elif test=='levene':
        T, pvalue = scipy.stats.levene(*samples, center=center, 
                             proportiontocut = proportiontocut)
    else:
        return
    print('========   TEST OF EQUAL VARIANCE   =======')
    print('%s test statistics T: %8.5f and P-vale %10.8f' % (test,T, pvalue))
    if pvalue<0.05:
        print('Reject null hypothesis at 95% confidence')
    else:
        print('Fail to reject null hypothesis at 95% confidence')
    return T, pvalue


def cochran_CUL(n, k, alpha):
    """
    helper function calculates upper limit critical value of
    test statistic of Cochran's C test
    
    args:
        n: number of data points per group
        k: number of groups
        alpha: significance level
    return:
        Tc: critical value of test statistic
    reference: https://en.wikipedia.org/wiki/Cochran's_C_test
    """
    Fc = scipy.stats.f.ppf(1-alpha/k, n-1, (n-1)*(k-1))
    return 1/(1 + (k-1)/Fc)

def cochranC(*args, alpha=0.05, plot=False, labels=None):
    """
    Cochran's C test for detecting outlier variance that is significantly
    larger than variances in other groups (an upper limit outlier test)
    Cochran's tests assumes balanced design, which means the size of samples
    in different groups must be equal.
    
    args:
        sample1, sample2, ....
        alpha: significance level, default=0.05
    
    return: 
        T: test statistic
        Tc: critical test statistic at confidence level specified by alpha
    """
    samples = np.vstack(args).T
    n, k    = samples.shape
    ss      = samples.var(ddof = 1, axis=0)
    T       = ss.max()/ss.sum()
    Tc      = cochran_CUL(n, k, alpha)
    
    print("=========  Cochran's C test  =========")
    print('# data per group  : %d'%(n))
    print('# of groups       : %d'%(k))
    print('Largest variance  : %f'%(ss.max()))
    print('Sum of variances  : %f'%(ss.sum()))
    print('Mean variance     : %f'%(ss.mean()))
    print('Significance level: %5.3f'%(alpha))
    print('Test statistic    : %f' %(T))
    print('Critical value    : %f' %(Tc))
    if T>Tc:
        print('Diagnostic: reject null')
    else:
        print('Diagnostic: fail to reject null')
    
    # make box plot.
    if not plot:
        return T, Tc
    
    # box plots.
    if not labels:
        labels = ['X%d'%(i) for i in range(k)] 
    if plot:
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.boxplot(args, positions=list(range(k)), labels=labels, widths=0.5)
        
        # plot a barplots of variances of different groups.
        plt.subplot(1,2,2)
        plt.bar(range(k), ss, color='red',
                alpha=0.8, align='center',tick_label=labels)
        plt.plot([-0.5, k-0.5], [np.mean(ss), np.mean(ss)], 'k--')
    
    return T, Tc
    
class PermTest():
    """
    A class that perform permutation test
    
    args:
        samples  : a list of samples
        stat_fun : a statistic function
    """
    def __init__(self, samples, stat_fun):
        self.samples      = samples
        self.stat_fun     = stat_fun
        self.T            = stat_fun(samples)
        self.distribution = None
        self.pvalue       = None
    def test(self, size = 1000, plot=False, bins=15):
        X = np.concatenate(self.samples)
        dims = list(map(len,self.samples))
        self.distribution = np.zeros((size,))
        for i in range(size):
            Y  = np.random.permutation(X)
            Ys = np.split(Y, np.cumsum(dims)[:-1])
            self.distribution[i] = self.stat_fun(Ys)
        self.pvalue    = np.mean(self.distribution>self.T)
        print('=============   Permutation Test   ============')
        print('%s test statistics T: %8.5f and P-vale %10.8f'
              % ('Permutation',self.T, self.pvalue))
        if self.pvalue<0.05:
            print('Reject null hypothesis at 95% confidence')
        else:
            print('Fail to reject null hypothesis at 95% confidence')
        
        # plot histogram
        if plot:
            plt.figure()
            plt.hist(self.distribution, alpha=0.8, bins = bins)
            
            
    
    