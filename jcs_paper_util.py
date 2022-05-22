import pandas as pd

import multiprocessing as mproc

import itertools as it

import tqdm
import time

import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import cycler
import matplotlib
import seaborn as sns

import ddsketch.ddsketch as ddsketch
import ddsketch.store as ddstore
import ddsketch.mapping as ddmapping

from random import randrange
from greedyks import GreedyKS
from IKS import IKS

distrib_types = ['normal', 'uniform', 'exp']

default_num_reps = 100

approx_methods = ['GreedyKS', 'Reservoir Sampling', 'IKS + RS', 'Lall + DDSketch']

def matplotlib_setup():
    matplotlib.rc('mathtext', fontset='cm')

    font = {
        'family':'serif', 
        'weight':'normal', 
        'size':8
    }
    matplotlib.rc('font', **font)

    legend = {
        'markerscale':1, 
        'labelspacing':0.3, 
        'borderpad':0.2, 
        'handletextpad':0.4,
        'columnspacing':1.2
    }
    matplotlib.rc('legend', **legend)

    axes_prop_cycle = {
        'markersize':[3]*20,
        #'markeredgewidth':[.8]*20,
        #'markeredgecolor':['w']*20,
        'linewidth':[1]*20,
        'linestyle':['-', '--', '-.', ':']*5,
        'marker':['o', 'X', 's', 'P', 'D']*4,
        'color':sns.color_palette("Set1", 20)
    }

    matplotlib.rc('axes', prop_cycle=cycler.cycler(**axes_prop_cycle))
    
def compute_ddsketch_error(sample, num_bins):
    range_limits = []
    
    if (sample > 0).any():
        range_limits.append([+sample[sample > 0].max(), +sample[sample > 0].min()])
        
    if (sample < 0).any():
        range_limits.append([-sample[sample < 0].min(), -sample[sample < 0].max()])
    
    alpha = float('-inf')
    
    for a, b in range_limits:
        x = (a/b)**(1/num_bins)
        alpha = max(alpha, (x - 1)/(x + 1))
    
    return alpha

def eval_error_D(ref_distrib, sample, num_bins):  
    exact_D = st.ks_1samp(sample, ref_distrib.cdf).statistic

    rs = ReservoirSampling(num_bins, ref_distrib)
    
    gks = GreedyKS(ref_distrib, num_bins, exact_prob=True)
    
    dds = LallDDSketch(compute_ddsketch_error(sample, num_bins), ref_distrib)
    
    iksr = IksReservoir(num_bins, ref_distrib)
    
    methods = {"Reservoir Sampling": rs,
               "GreedyKS": gks,
               "Lall + DDSketch": dds,
               "IKS + RS": iksr,
              }
    
    for observation in sample:
        for m in methods:
            methods[m].add_element(observation)
    
    error_D = {}
    
    for m in methods:
        error_D[m] = abs(methods[m].get_D() - exact_D)
        
    return error_D

#@profile
def eval_effectiveness(ref_distrib, sample, num_bins):
    drift_point = len(sample)//10
    sample = sample[:int(len(sample)*.9)]
    stream = np.concatenate((ref_distrib.rvs(drift_point), sample))

    rs = ReservoirSampling(num_bins, ref_distrib)
    
    eks = ReservoirSampling(len(stream), ref_distrib)
    
    gks = GreedyKS(ref_distrib, num_bins, exact_prob=True)
    
    dds = LallDDSketch(compute_ddsketch_error(sample, num_bins), ref_distrib)
    
    iksr = IksReservoir(num_bins, ref_distrib)
    
    methods = {"Reservoir Sampling": {'method':rs, 'stop':False},
               "Exact KS": {'method':eks, 'stop':False},
               "GreedyKS": {'method':gks, 'stop':False},
               "Lall + DDSketch": {'method':dds, 'stop':False},
               "IKS + RS": {'method':iksr, 'stop':False},
              }
    
    respon = {"Reservoir Sampling": float('nan'),
              "Exact KS": float('nan'),
              "GreedyKS": float('nan'), 
              "Lall + DDSketch": float('nan'), 
              "IKS + RS": float('nan'),
             }
   
    for t, element in enumerate(stream):
        for m in methods:
            if not methods[m]['stop']:
                methods[m]['method'].add_element(element)
                if t >= num_bins and methods[m]['method'].detected_change():
                    respon[m] = t - drift_point
                    methods[m]['stop'] = True
                    
    return respon

def eval_efficiency(ref_distrib, sample, num_bins, interval):
        
    rs = ReservoirSampling(num_bins, ref_distrib)
    
    gks = GreedyKS(ref_distrib, num_bins, exact_prob=True)
    
    dds = LallDDSketch(compute_ddsketch_error(sample, num_bins), ref_distrib)
    
    iksr = IksReservoir(num_bins, ref_distrib)
    
    methods = {
            "Reservoir Sampling": rs,
            "GreedyKS": gks,
            "Lall + DDSketch": dds,
            "IKS + RS": iksr,
            }
    
    eff = {}
    
    for m in methods:
        begin = time.time()
        for t, observation in enumerate(sample):
            methods[m].add_element(observation)
            if t % interval == 0:
                methods[m].get_D()
        eff[m] = time.time() - begin

    return eff

def eval_eda(ref_distrib, sample, num_bins):
    drift_point = len(sample)//10
    sample = sample[:int(len(sample)*.9)]
    stream = np.concatenate((ref_distrib.rvs(drift_point), sample))

    rs = ReservoirSampling(num_bins, ref_distrib)
    
    eks = ReservoirSampling(len(stream), ref_distrib)
    
    gks = GreedyKS(ref_distrib, num_bins, exact_prob=True)
    
    dds = LallDDSketch(compute_ddsketch_error(sample, num_bins), ref_distrib)
    
    iksr = IksReservoir(num_bins, ref_distrib)
    
    methods = {"Reservoir Sampling": {'method':rs, 'stop':False},
               "Exact KS": {'method':eks, 'stop':False},
               "GreedyKS": {'method':gks, 'stop':False},
               "Lall + DDSketch": {'method':dds, 'stop':False},
               "IKS + RS": {'method':iksr, 'stop':False},
              }
    
    respon = {"Reservoir Sampling": [],
              "Exact KS": [],
              "GreedyKS": [], 
              "Lall + DDSketch": [], 
              "IKS + RS": [],
             }
   
    for t, element in enumerate(stream):
        for m in methods:
            if not methods[m]['stop']:
                methods[m]['method'].add_element(element)
                if t >= num_bins and methods[m]['method'].detected_change():
                    respon[m].append(t - drift_point)
                    methods[m]['stop'] = True
                    
    return respon
    
def gen_test_distribs(mean_diff, std_diff, type_):
    mean = std = 1
    
    if type_ == 'normal':
        distrib = st.norm
        
        params = [
            [mean, std], 
            [mean + mean_diff, std + std_diff]
        ]
        
    elif type_ == 'uniform':
        distrib = st.uniform

        params = [
            [mean - std*12**.5/2, std*12**.5], 
            [mean + mean_diff - (std+std_diff)*12**.5/2, (std+std_diff)*12**.5]
        ]
        
    elif type_ == 'exp':
        distrib = st.expon
    
        params = [
            [mean-std, std],
            [(mean+mean_diff) - (std+std_diff), std+std_diff]
        ]
    else:
        raise Exception('Unknown distribution')
    
    return distrib(*params[0]), distrib(*params[1])

def plot_log_error_bars(data, axis, exp_type, legend=False):
    plt.sca(axis)
    
    #display(data)

    data = data.groupby(exp_type)[approx_methods]

    all_means = data.mean()    
    all_stds = data.std()

    for algorithm in all_means:
        mean = all_means[algorithm]
        
        plt.plot(mean.index, mean, label=algorithm if legend else None)

        std = all_stds[algorithm]

        relative_error = (mean+std)/mean

        y1 = np.array(mean*relative_error)
        #y1 = [i[0] for i in y1]

        y2 = np.array(mean/relative_error)
        #y2 = [i[0] for i in y2]

        plt.fill_between(mean.index, y1, y2, alpha=.2)
        
    if legend:
        plt.legend()
        
def plot_errors2(results, y_label, fig_name, ylog=True):
    # results => type:(DataFrame,'label')
    
    fig = plt.figure(figsize=(7.,2.))
    gs = fig.add_gridspec(1,3, wspace=0.)

    axs = gs.subplots(sharey=True)
    
    for i, type_ in enumerate(results):
        df = results[type_][0]
        data = df[df.distrib == 'normal']
        
        axs[i].set_xmargin(0.1)
    
        plot_log_error_bars(data, axs[i], type_, i == 1)
        
        if ylog:
            axs[i].set(xscale="log", yscale="log")
        else:
            axs[i].set(xscale="log", yscale="linear")
        
        if i > 0:
            axs[i].tick_params(axis='y', which='both', left=False, right=False)
        
        axs[i].set(xlabel=results[type_][1])
    
    axs[0].set(ylabel=y_label)
    axs[1].set(title='Efficiency assessment, Normal distribution')
    axs[1].legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.45))
  
    plt.savefig(fig_name, bbox_inches='tight')
    
        
def plot_errors(result, titles, y_label, x_label, exp_type, fig_name, ylog=True):
    fig = plt.figure(figsize=(7.,2.))
    gs = fig.add_gridspec(1,3, wspace=0.)

    axs = gs.subplots(sharey=True)
    
    for i, type_ in enumerate(distrib_types):
        data = result[result.distrib == type_]
        
        axs[i].set_xmargin(0.1)
    
        plot_log_error_bars(data, axs[i], exp_type, i == 1)
        
        if ylog:
            axs[i].set(title=titles[i], xscale="log", yscale="log")
        else:
            axs[i].set(title=titles[i], xscale="log", yscale="linear")
        
        if i > 0:
            axs[i].tick_params(axis='y', which='both', left=False, right=False)
    
    axs[0].set(ylabel=y_label)
    axs[1].legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.4))
    
    

    fig.text(0.5, -0.045, x_label, ha="center")
    
    plt.savefig(fig_name, bbox_inches='tight')
    
def format_table(x):
    return '{:d}/{:d}/{:.1f}'.format((x < 0).sum(), x.isna().sum(), x[x >= 0].mean())
    
def plot_table(result, exp_type, tex_name=None):
  
    tables = {}
    
    for distrib in distrib_types:
        table = pd.DataFrame()
        for k, v in result[result.distrib == distrib].groupby(exp_type):
            table[k] = pd.DataFrame(v[approx_methods].apply(format_table))

        table = table.rename(columns={0.001: '$10^{-3}$',
                                      10**-2.5:'$10^{-2.5}$',
                                      10**-1.5:'$10^{-1.5}$',
                                      0.01:'$10^{-2}$',
                                      0.1:'$10^{-1}$',
                                      10**1:'$10^1$',
                                      10**2:'$10^2$',
                                      10**3:'$10^3$',
                                      10**4:'$10^4$',
                                      3162:'$10^{3.5}$',
                                      31:'$10^{1.5}$',
                                      
                                     }).T
        
        tex_file = tex_name + '_' + distrib + '.tex' if tex_name else None
        table.style.to_latex(tex_file)
        
        tables[distrib] = table
    
    return tables
    
def args_gen(
    sample_size=(10**4,), 
    memo_units=(10**2,), 
    mean_diff=(0.1,),
    std_diff=(0.0,),
    q_interval=(1,)):
    
    for nr in range(default_num_reps):
        yield from it.product(sample_size, memo_units, mean_diff, std_diff, q_interval, distrib_types)
        
def single_process(args):
    ss, mu, md, sd, qi, type_ = args
    ref_distrib, alt_distrib = gen_test_distribs(md, sd, type_)
    sample = alt_distrib.rvs(ss)
    num_bins = mu
    error_D = eval_error_D(ref_distrib, sample, num_bins)
    error_D['distrib'] = type_
    error_D['memo_units'] = mu
    error_D['sample_size'] = ss
    error_D['mean_diff'] = md
    error_D['std_diff'] = sd
    
    return error_D

def single_process_eff(args):
    ss, mu, md, sd, qi, type_ = args
    ref_distrib, alt_distrib = gen_test_distribs(md, sd, type_)
    sample = alt_distrib.rvs(ss)
    num_bins = mu
    eff = eval_efficiency(ref_distrib, sample, num_bins, qi)
    eff['distrib'] = type_
    eff['memo_units'] = mu
    eff['sample_size'] = ss
    eff['mean_diff'] = md
    eff['std_diff'] = sd
    eff['q_interval'] = qi
    
    return eff

def get_results(nproc=None, **args_gen_args):
    results = mproc.Pool(processes=nproc).imap(single_process, args_gen(**args_gen_args))
    total_tests = default_num_reps * len(next(iter(args_gen_args.values()))) * len(distrib_types)
    results = tqdm.tqdm(results, total=total_tests)
    return pd.DataFrame(results)

class ReservoirSampling():
    def __init__(self, size, ref_distrib, p_threshold=0.01):
        self.size = size
        self.ref_distrib = ref_distrib
        self.p_threshold = p_threshold
        self.reset()

    def reset(self):
        self.reservoir = []
        self.t = 0
        
    #@profile
    def add_element(self, element):
        if len(self.reservoir) < self.size:
            self.reservoir.append(element)
        else:
            r = randrange(0, self.t)
            if r < self.size:
                self.reservoir[r] = element
        self.t += 1
        
    def get_D(self):
        return np.abs(self.ref_distrib.cdf(np.sort(self.reservoir)) - np.linspace(0,1,len(self.reservoir))).max()
    
    def detected_change(self):
        return st.kstwo.sf(self.get_D(), len(self.reservoir)) <= self.p_threshold

class IksReservoir():
    def __init__(self, size, ref_distrib, p_threshold=0.01):
        self.size = size
        self.ref_distrib = ref_distrib
        self.p_threshold = p_threshold
        self.reset()

    def reset(self):
        self.iks = IKS()
        self.reservoir = [[], []]
        self.t = 0

    #@profile
    def add_element(self, element):
        if len(self.reservoir[0]) < self.size:
            ref_element = self.ref_distrib.rvs(1)
        
            self.reservoir[0].append(ref_element)
            self.iks.Add(ref_element, 0)
            
            self.reservoir[1].append(element)
            self.iks.Add(element, 1)
        else:
            r = randrange(0, self.t)
            if r < self.size:
                ref_element = self.ref_distrib.rvs(1)
        
                self.iks.Remove(self.reservoir[0][r], 0)
                self.reservoir[0][r] = ref_element
                self.iks.Add(ref_element, 0)
                
                self.iks.Remove(self.reservoir[1][r], 1)
                self.reservoir[1][r] = element
                self.iks.Add(element, 1)
        
        self.t += 1
        
    def get_D(self):
        return self.iks.KS()
    
    def detected_change(self):
        ca = self.iks.CAForPValue(self.p_threshold)
        return self.iks.Test(ca)

class LallDDSketch():
    def __init__(self, error, ref_distrib, p_threshold=0.01):
        self.error = error
        self.p_threshold = p_threshold
        self.ref_distrib = ref_distrib
        self.reset()
        
    def reset(self):
        self.dds = ddsketch.BaseDDSketch(
            ddmapping.LogarithmicMapping(self.error),
            ddstore.DenseStore(1),
            ddstore.DenseStore(1),
            0
        )
        
        self.vec_get_quantile_value = np.vectorize(self.dds.get_quantile_value)
        self.percents = []
    
    #@profile
    def add_element(self, element):
        self.dds.add(element)
    
    
    #@profile
    def get_D(self):
        num_bins = len(self.dds.store.bins) + len(self.dds.negative_store.bins)
        if num_bins != len(self.percents):
            self.percents = np.linspace(0.0, 1.0, num_bins)
        quantile_values = self.vec_get_quantile_value(self.percents)
        cdf_values = self.ref_distrib.cdf(quantile_values)
        dds_D = np.abs(cdf_values - self.percents).max()
        return dds_D
    
    
#    #@profile
#    def get_D(self):
#        dds_D = 0
#        num_bins = len(self.dds.store.bins) + len(self.dds.negative_store.bins)
#        for percent in np.linspace(0.0, 1.0, num_bins):
#            v = self.dds.get_quantile_value(percent)
#            cdf_v = self.ref_distrib.cdf(v)
#            dds_D = max(dds_D, abs(percent-cdf_v))
#        return dds_D
    
    
    def detected_change(self):
        return st.kstwo.sf(self.get_D(), self.dds.count) <= self.p_threshold

#class LallDDSketch():
#    def __init__(self, num_bins, error, ref_distrib, p_threshold=0.01):
#        self.error = error
#        self.p_threshold = p_threshold
#        self.ref_distrib = ref_distrib
#        self.percents = np.linspace(0.0, 1.0, num_bins)
#        self.reset()
#        
#    def reset(self):
#        self.dds = []
#        self.vec_get_quantile_value = []
#        
#        for i in range(2):
#            dds.append(ddsketch.BaseDDSketch(
#                ddmapping.LogarithmicMapping(self.error),
#                ddstore.DenseStore(1),
#                ddstore.DenseStore(1),
#                0
#            ))
#            self.vec_get_quantile_value.append(np.vectorize(self.dds[-1].get_quantile_value))
#    
#    @profile
#    def add_element(self, element):
#        self.dds[0].add(element)
#        self.dds[1].add(self.ref_distrib.rvs(1))
#    
#    @profile
#    def get_D(self):
#        quantile_values = self.vec_get_quantile_value(self.percents)
#        cdf_values = self.ref_distrib.cdf(quantile_values)
#        dds_D = np.abs(cdf_values - self.percents).max()
#        return dds_D    
#    
#    @profile
#    def detected_change(self):
#        num_bins = len(self.dds.store.bins) + len(self.dds.negative_store.bins)
#        D = self.get_D()
#        prob = st.kstwobign.sf(D * self.dds.count**.5)
#        return (prob <= self.p_threshold)
