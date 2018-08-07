#!/usr/bin/env python

# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# August 2018

"""
Simulate naive GWAS.
"""

import argparse
import logging
import numpy as np
import os
import pandas as pd 
import sys

sigmoid = lambda x: 1./(1. + np.exp(-x))

def generate_data(num_feats, num_obsvs, num_causl, binary):
    """
    Generate data with the given parameters.

    Arguments
    ---------
    num_feats: int
        Number of SNPs.

    num_obsvs: int
        Number of samples.

    num_causl: int
        Number of causal SNPs,

    binary: boolean
        Whether or not to create a binary phenotype.


    Returns
    -------
    X: (num_obsvs, num_feats) int numpy array
        Binary SNPs.
    
    y: (num_obsvs, ) numpy array
        Phenotypes.
    
    causl: list of int
        Indices of the causal SNPs.
    
    w_causl: (num_causl, ) numpy array
        Coefficients of the causal SNPs.
    """
    # SNPs
    X = np.random.binomial(1, 0.1, size=(num_obsvs, num_feats))

    # Phenotype
    w_causl = np.random.normal(loc=0.2, scale=0.05, size=(num_causl))
    
    w = np.zeros((num_feats, ))
    w[:num_causl] = w_causl
    
    y = np.dot(X, w) + np.random.normal(loc=0., scale=0.1, size=(num_obsvs, ))
    if binary:
        # Apply a logit transform + threshold
        y = np.where(sigmoid(y) > 0.5, 1, 0)
    
    # Shuffle
    map_indices_l = range(num_feats)
    np.random.shuffle(map_indices_l)
    map_indices = dict(zip(range(num_feats), map_indices_l))
    map_indices_r = dict(zip(map_indices_l, range(num_feats)))

    X = X[:, map_indices_l]
    causl = [map_indices_r[ix] for ix in range(num_causl)]

    return X, y, causl, w_causl


def save_data(X, y, causl, w_causl, data_rep):
    """
    Save the generated data in files with appropriate formats.

    Arguments:
    ----------
    X: (num_obsvs, num_feats) int numpy array
        Binary SNPs.
    
    y: (num_obsvs, ) numpy array
        Phenotypes.
    
    causl: list of int
        Indices of the causal SNPs.
    
    w_causl: (num_causl, ) numpy array
        Coefficients of the causal SNPs.

    data_rep: file path
        Repository where to save the data

    Creates:
    -------
    <data_rep>/X.data
        path to the white-space separated file containing the genotypes
        as a matrix with as many lines as samples and as many columns as SNPs
        each entry in the matrix is the SNP binary encoding.
        Each line starts with a genome identifier.
        Each column starts with a SNP identifier.    
        
    <data_rep>/y.data
        path to file containing the phenotypes
        one line per sample, containing one value.
        Each line starts with a genome identifier.
        The column starts with a phenotype identifier.

    <data_rep>/causl.data
        path to file containing the name of the causal SNPs.

    <data_rep>/w_causl.data
        path to file containing the weights assigned to the causal SNPs.    
    """
    # Create file names
    X_fname = '%s/X.data' % data_rep
    y_fname = '%s/y.data' % data_rep
    causl_fname = '%s/causl.data' % data_rep
    wghts_fname = '%s/w_causl.data' % data_rep
    
    # Create data repository
    if not os.path.exists(data_rep):
        logging.debug("Creating %s" % data_rep)
        os.mkdir(data_rep)
    
    # Save genotype data
    df = pd.DataFrame(X)
    df.index = ['Sample_%d' % ix for ix in range(X.shape[0])]
    df.to_csv(X_fname, sep=' ',
              header=['SNP_%d' % ix for ix in range(X.shape[1])],
              index=True)
    del df

    # Save phenotype data
    df = pd.DataFrame(y)
    df.index = ['Sample_%d' % ix for ix in range(y.shape[0])]
    df.to_csv(y_fname, sep=' ', header=['Phenotype'], index=True)
    del df

    # Save causal SNP data
    np.savetxt(causl_fname, causl, fmt='%d')
    np.savetxt(wghts_fname, w_causl)


def main():
    """
    Generate GWAS data naively: the SNPs follow a binomial distribution and the phenotype
    is a function of a linear combination of the causal SNPs (+ some noise).

    Usage example
    -------------
    python simulate_naive_gwas.py 1000 150 10 ../../data/simu1 --binary

    Will generate
    ../../data/simu1/X.data
        path to the white-space separated file containing the genotypes
        as a matrix with as many lines as samples and as many columns as SNPs
        each entry in the matrix is the SNP binary encoding.
        Each line starts with a genome identifier.
        Each column starts with a SNP identifier.    
        
    ../../data/simu1/y.data
        path to file containing the phenotypes
        one line per sample, containing one value.
        Each line starts with a genome identifier.
        The column starts with a phenotype identifier.

    ../../data/simu1/causl.data
        path to file containing the name of the causal SNPs.

    ../../data/simu1/w_causl.data
        path to file containing the weights assigned to the causal SNPs.    
    """
    parser = argparse.ArgumentParser(description="Generate GWAS data naively", add_help=True)
    parser.add_argument("num_feats", help="Number of SNPs", type=int)
    parser.add_argument("num_obsvs", help="Number of samples", type=int)
    parser.add_argument("num_causl", help="Number of causal SNPs", type=int)
    parser.add_argument("data_rep", help="Repository where to save the data", type=str)
    parser.add_argument("-b", "--binary", dest='binary', help="Generate a binary phenotype",
                        action='store_true')
    args = parser.parse_args()

    X, y, causl, w_causl = generate_data(args.num_feats, args.num_obsvs, args.num_causl, args.binary)
    save_data(X, y, causl, w_causl, args.data_rep)
    

if __name__ == "__main__":
    main()
