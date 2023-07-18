# -*- coding: utf-8 -*-
"""
MCMC: analyze chain
""" 

import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from multiprocessing import Pool
from chainconsumer import ChainConsumer

from mcmc_pic import pic_chain, pic_fit


def mcmc_analyze(sampler, data, model, init, prior_data, amputate, params_names, 
                 **kwargs):
    # args management
    x, y, yerr, *_ = *data, None
    ndim = len(init) 

    # kwargs
    prnt = kwargs['prnt'] if 'prnt' in kwargs else False
    show = kwargs['show'] if 'show' in kwargs else False
    save = kwargs['save'] if 'save' in kwargs else False

    flat_chain = sampler.get_chain(discard=amputate, flat=True)
    c = ChainConsumer()
    c.add_chain(flat_chain, parameters=params_names)
    summary = c.analysis.get_summary()

    # print
    if prnt:
        s = [" {:>4}: {}".format(k, summary[k]) for k in summary.keys()]
        print("\nMCMC results:", *s, sep='\n')

    # pics
    if show or save:
        fig0 = c.plotter.plot(legend=False, figsize=(6, 6))
        fig1 = pic_chain(sampler, amputate=amputate, params_names=params_names)
        fig2 = pic_fit(sampler, model, data, prior_data)

        #fig1 = c.plotter.plot_walks(convolve=100, figsize=(6, 6))
        if save:
            fig0.savefig(save[0], bbox_inches='tight')
            fig1.savefig(save[1], bbox_inches='tight')
            fig2.savefig(save[2], bbox_inches='tight')

            # garved collector
            plt.close('all')
            gc.collect()
        if show:
            plt.show()

    return summary

