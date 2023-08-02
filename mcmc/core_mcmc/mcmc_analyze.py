# -*- coding: utf-8 -*-
"""
MCMC: analyze chain
""" 

import numpy as np
from chainconsumer import ChainConsumer

def mcmc_analyze(sampler, amputate=0, params_names=None):
    flat_chain = sampler.get_chain(discard=amputate, flat=True)
    c = ChainConsumer()
    c.add_chain(flat_chain, parameters=params_names)
    return c

def mcmc_summary(c, prnt=False):
    summary = c.analysis.get_summary()

    # print
    if prnt:
        s = [" {:>4}: {}".format(k, summary[k]) for k in summary.keys()]
        print("\nMCMC results:", *s, sep='\n')

    return summary

