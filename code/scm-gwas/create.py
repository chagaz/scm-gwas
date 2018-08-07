#!/usr/bin/env python

# @Author 
# Chloe-Agathe Azencott
# chloe-agathe.azencott@mines-paristech.fr
# August 2018

"""
Create a HDF5 file for processing by the Set Covering Machine code.
"""

import argparse
import h5py # as h 
import gc
import logging
import numpy as np
import pandas as pd
from progressbar import Bar, Percentage, ProgressBar, Timer

import sys
from time import time
from uuid import uuid1

from utils import _minimum_uint_size, _pack_binary_bytes_to_ints

SNP_MATRIX_PACKING_SIZE = 64
SNP_MATRIX_DTYPE = np.uint64
PHENOTYPE_LABEL_DTYPE = np.uint8
BLOCK_SIZE = 100000


def _create_hdf5_file_no_chunk_caching(path):
    """
    Create a HDF5 file.
    Inspired from kover/core/kover/create.py 

    Arguments:
    ---------
    path: file path
        Path of the HD5F file to create

    Returns:
    --------
    h5py_file: HDF5 file
    
    """
    # Create the HDF5 File
    h5py_file = h5py.File(path, "w")

    # Get the default cache properties
    access_property_list = h5py_file.fid.get_access_plist() 
    cache_properties = list(access_property_list.get_cache())
    h5py_file.close()

    # Disable chunk caching
    cache_properties[2] = 0  # No chunk caching
    access_property_list.set_cache(*cache_properties)
    file_id = h5py.h5f.open(bytes(path, 'utf-8'), # convert str to bytes for Python3
                            flags=h5py.h5f.ACC_RDWR, # open in read-write mode
                            fapl=access_property_list)

    # Reopen the file without a chunk cache
    h5py_file = h5py.File(file_id)

    return h5py_file


def from_tsv(tsv_path, output_path, phenotype_name, phenotype_path, gzip,
             progress_callback=None):
    """
    Create HDF5 file from a white-space-separated file.
    Inspired from kover/core/kover/dataset/create.py    

    tsv_path: file path
        path to the white-space separated file containing the genotypes
        as a matrix with as many lines as samples and as many columns as SNPs
        each entry in the matrix is the SNP binary encoding.
        Each line starts with a genome identifier.
        Each column starts with a SNP identifier.    

    output_path: file path
        path to the HDF5 file to create.

    phenotype_name: str
        name of the phenotype.

    phenotype_path: file path
        path to file containing the phenotypes
        one line per sample, containing one value.
        Each line starts with a genome identifier.
        The column starts with a phenotype identifier.

    gzip: int
         gzip compression level to use in the HD5F file.

    progress_callback: execution callback function
    """
    # Execution callback functions
    warning_callback = lambda w: logging.warning(w)

    def normal_raise(exception):
        raise exception
    error_callback = normal_raise

    if progress_callback is None:
        progress_callback = lambda t, p: None

    # Compression
    compression = "gzip" if gzip > 0 else None
    compression_opts = gzip if gzip > 0 else None

    # Create h5py file and set meta-data attributes
    h5py_file = _create_hdf5_file_no_chunk_caching(output_path)
    h5py_file.attrs["created"] = time()
    h5py_file.attrs["uuid"] = str(uuid1()) # universally unique identifier
    h5py_file.attrs["genome_source_type"] = "tsv"
    h5py_file.attrs["genomic_data"] = tsv_path
    h5py_file.attrs["phenotype_name"] = phenotype_name if phenotype_name is not None else "NA"
    h5py_file.attrs["phenotype_data"] = phenotype_path 
    h5py_file.attrs["compression"] = "gzip (level %d)" % gzip

    # Read list of genome and SNP identifiers
    reader = pd.read_table(tsv_path,
                           sep='\s+', # whitespace separator
                           index_col=0, # columns to use for row labels
                           iterator=True, engine="c")
    # genome_ids = []
    # for chunk in reader:
    #     genome_ids.extend(list(chunk.index))
    # snp_ids = list(chunk.columns.values)
    snp_ids = []
    for chunk in reader:
        snp_ids.extend(list(chunk.index))
    genome_ids = list(chunk.columns.values)
    num_snps = len(snp_ids)
    del reader
    logging.debug("The SNP matrix contains %d genomes." % len(genome_ids))
    if len(set(genome_ids)) < len(genome_ids):
        error_callback(Exception("The genomic data contains multiple genomes with the same identifier."))

    # Write SNP information
    # in Python3, need to convert to binary string
    logging.debug("Creating the SNP identifier dataset.")
    h5py_file.create_dataset("snp_identifiers",
                             data=[bytes(sid, 'utf-8') for sid in snp_ids])  
    
    # Read phenotype information + sort samples by label for optimal performance.
    reader = pd.read_table(phenotype_path,
                           sep='\s+', # whitespace separator
                           index_col=0, # columns to use for row labels
                           )
    #labels = list(reader.index)
    labels = reader.values
    labels = labels.reshape(labels.shape[0], )
    del reader
    logging.debug("Sorting samples by label for optimal performance.")
    sorter = np.argsort(labels)
    genome_ids = [genome_ids[ix] for ix in sorter]
    labels = [labels[ix] for ix in sorter]

    # Write phenotype information
    logging.debug("Creating the phenotype dataset.")
    phenotype = h5py_file.create_dataset("phenotype", data=labels, dtype=PHENOTYPE_LABEL_DTYPE,
                                         compression=compression,
                                         compression_opts=compression_opts)
    phenotype.attrs["name"] = phenotype_name
    del phenotype, labels
        
    # Write sorted genome ids
    # in Python3, need to convert to binary string
    logging.debug("Creating the genome identifier dataset.")
    h5py_file.create_dataset("genome_identifiers",
                             data=[bytes(gid, 'utf-8') for gid in genome_ids],
                             compression=compression,
                             compression_opts=compression_opts)

    # Initialize snp_matrix dataset
    logging.debug("Creating the SNP matrix dataset.")
    num_rows = int(np.ceil(1.0 * len(genome_ids) / SNP_MATRIX_PACKING_SIZE))
    tsv_block_size = min(num_snps, BLOCK_SIZE)

    snp_matrix = h5py_file.create_dataset("snp_matrix",
                                           shape=(num_rows, num_snps),
                                           dtype=SNP_MATRIX_DTYPE,
                                           compression=compression,
                                           compression_opts=compression_opts,
                                           chunks=(1, tsv_block_size))

    logging.debug("Transferring the data from TSV to HDF5.")
    tsv_reader = pd.read_table(tsv_path,
                           sep='\s+', # whitespace separator
                           index_col=0, # columns to use for row labels
                            chunksize=tsv_block_size)
    n_blocks = int(np.ceil(1.0 * num_snps / tsv_block_size))
    n_copied_blocks = 0.
    for i, chunk in enumerate(tsv_reader):
        logging.debug("Block %d/%d." % (i + 1, n_blocks))
        progress_callback("Creating", n_copied_blocks / n_blocks)
        logging.debug("Reading data from TSV file.")
        read_block_size = chunk.index.shape[0]
        block_start = i * tsv_block_size
        block_stop = block_start + read_block_size
        n_copied_blocks += 0.5
        progress_callback("Creating", n_copied_blocks / n_blocks)

        if block_start > num_snps:
            break

        logging.debug("Writing data to HDF5.")
        data_chunk = chunk[genome_ids].T.values.astype(np.uint8)
        logging.debug("Packing the data.")
        print(snp_matrix[:, block_start:block_stop].shape)
        print(_pack_binary_bytes_to_ints(data_chunk, pack_size=SNP_MATRIX_PACKING_SIZE).shape)
                                                                           
        snp_matrix[:, block_start:block_stop] = _pack_binary_bytes_to_ints(data_chunk,
                                                                           pack_size=SNP_MATRIX_PACKING_SIZE)
        n_copied_blocks += 0.5
        progress_callback("Creating", n_copied_blocks / n_blocks)

        logging.debug("Garbage collection.")
        gc.collect()  

    h5py_file.close()

    logging.debug("Dataset creation completed.")


def main():
    """
    Create HDF5 data set from genotype and phenotype data.

    Usage example
    -------------
    python create.py --from-tsv --genomic-data ../../data/simu1/X.data --phenotype-name Phenotype \
    --phenotype-data ../../data/simu1/y.data --output ../../data/simu1/data.hdf5 --verbose

    Will create the HDF5 file ../../data/simu1/data.hdf5.
    """
    parser = argparse.ArgumentParser(description="Create HDF5 data set from genotype and phenotype data.",
                                     add_help=True)
    # parser.add_argument("", help="", type=int)
    # parser.add_argument("-", "--", default=0, help="", type=int)
    parser.add_argument("--from-tsv", dest='from_tsv', help="From a tsv file",
                        action='store_true')

    parser.add_argument('--genomic-data', help='A genomic data file.',
                        required=True)
    parser.add_argument('--phenotype-name', help='The phenotype name.')
    parser.add_argument('--phenotype-data', help='A phenotypic data file.', required=True)

    parser.add_argument('--output', help='The HDF5 dataset to be created.', required=True)
    parser.add_argument('--compression', type=int,
                        help='The gzip compression level (0 - 9). 0 means no compression'
                        '. The default value is 4.', default=4)
    parser.add_argument('-x', '--progress', help='Shows a progress bar for the execution.',
                        action='store_true')
    parser.add_argument('-v', '--verbose', help='Sets the verbosity level.', default=False,
                        action='store_true')

    # If no argument has been specified, default to help
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    # Parse arguments
    args = parser.parse_args()

    # Input validation logic
    if not args.from_tsv:
        print("Error: For now the only input type implemented is TSV.")
        exit()
        
    # Verbosity
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s.%(msecs)d %(levelname)s %(module)s - " + \
                            "%(funcName)s: %(message)s")
    # Progress bars
    if args.progress:
        progress_vars = {"current_task": None, "pbar": None}

        def progress(task_name, p):
            if task_name != progress_vars["current_task"]:
                if progress_vars["pbar"] is not None:
                    progress_vars["pbar"].finish()
                progress_vars["current_task"] = task_name
                progress_vars["pbar"] = ProgressBar(widgets=['%s: ' % task_name, \
                                                             Percentage(), Bar(), Timer()],
                                                    maxval=1.0)
                progress_vars["pbar"].start()
            else:
                progress_vars["pbar"].update(p)
    else:
        progress = None

    from_tsv(tsv_path=args.genomic_data,
             output_path=args.output,
             phenotype_name=args.phenotype_name,
             phenotype_path=args.phenotype_data,
             gzip=args.compression,
             progress_callback=progress)

    if args.progress:
        progress_vars["pbar"].finish()

if __name__ == "__main__":
    main()