#!/usr/bin/env python

# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# August 2018

"""
Utilities for scm-gwas.
"""

import argparse
import numpy as np
import os
import sys


def _minimum_uint_size(max_value):
    """
    Find the minimum size unsigned integer type that can store values of at most max_value.
    Copied from kover/core/kover/utils.py
    """
    if max_value <= np.iinfo(np.uint8).max:
        return np.uint8
    elif max_value <= np.iinfo(np.uint16).max:
        return np.uint16
    elif max_value <= np.iinfo(np.uint32).max:
        return np.uint32
    elif max_value <= np.iinfo(np.uint64).max:
        return np.uint64
    else:
        return np.uint128

def _pack_binary_bytes_to_ints(a, pack_size):
    """
    Packs binary values stored in bytes into ints
    Copied from kover/core/kover/utils.py    
    """
    if pack_size == 64:
        type = np.uint64
    elif pack_size == 32:
        type = np.uint32
    else:
        raise ValueError("Supported data types are 32-bit and 64-bit integers.")

    b = np.zeros((int(np.ceil(1.0 * a.shape[0] / pack_size)), a.shape[1]), dtype=type)
    packed_rows = 0
    packing_row = 0
    for i in range(a.shape[0]): # replace xrange with range in Python3
        if packed_rows == pack_size:
            packed_rows = 0
            packing_row += 1
        tmp = np.asarray(a[i], dtype=type)
        tmp = np.left_shift(tmp, type(pack_size - packed_rows - 1))
        np.bitwise_or(b[packing_row], tmp, out=b[packing_row])
        packed_rows += 1

    return b


def _unpack_binary_bytes_from_ints(a):
    """
    Unpacks binary values stored in bytes into ints
    Copied from kover/core/kover/utils.py    
    """
    type = a.dtype

    if type == np.uint32:
        pack_size = 32
    elif type == np.uint64:
        pack_size = 64
    else:
        raise ValueError("Supported data types are 32-bit and 64-bit integers.")

    unpacked_n_rows = a.shape[0] * pack_size
    unpacked_n_columns = a.shape[1] if len(a.shape) > 1 else 1
    b = np.zeros((unpacked_n_rows, a.shape[1]) if len(a.shape) > 1 else unpacked_n_rows, dtype=np.uint8)

    packed_rows = 0
    packing_row = 0
    for i in xrange(b.shape[0]):
        if packed_rows == pack_size:
            packed_rows = 0
            packing_row += 1
        tmp = np.left_shift(np.ones(unpacked_n_columns, dtype=type), pack_size - (i - pack_size * packing_row)-1)
        np.bitwise_and(a[packing_row], tmp, tmp)
        b[i] = tmp > 0
        packed_rows += 1

    return b

    
    

