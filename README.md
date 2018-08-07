# scm-gwas
Using set covering machines for genome-wide association studies.

# Requirements
Code developed with
* Python 3.4.3
* Numpy 1.15.0
* Scipy 1.1.0
* Pandas 0.22.0
* h5py 2.8.0

# Input data file specification
```
HDF5 "example.hdf5" {
FILE_CONTENTS {
 group      /
    dataset    /genome_identifiers
    dataset    /snp_identifiers
    dataset    /snp_matrix
    dataset    /phenotype
    group      /splits
       group      /splits/my_split_name_1
          dataset    /splits/my_split_name_1/test_genome_idx
          dataset    /splits/my_split_name_1/train_genome_idx
          dataset    /splits/my_split_name_1/unique_risk_by_anti_kmer
          dataset    /splits/my_split_name_1/unique_risk_by_kmer
          dataset    /splits/my_split_name_1/unique_risks
       group      /splits/my_split_name_2
          dataset    /splits/my_split_name_2/test_genome_idx
          dataset    /splits/my_split_name_2/train_genome_idx
          dataset    /splits/my_split_name_2/unique_risk_by_anti_kmer
          dataset    /splits/my_split_name_2/unique_risk_by_kmer
          dataset    /splits/my_split_name_2/unique_risks
          group      /splits/my_split_name_2/folds
             group      /splits/my_split_name_2/folds/fold_1
                dataset    /splits/my_split_name_2/folds/fold_1/test_genome_idx
                dataset    /splits/my_split_name_2/folds/fold_1/train_genome_idx
                dataset    /splits/my_split_name_2/folds/fold_1/unique_risk_by_anti_kmer
                dataset    /splits/my_split_name_2/folds/fold_1/unique_risk_by_kmer
                dataset    /splits/my_split_name_2/folds/fold_1/unique_risks
             group      /splits/my_split_name_2/folds/fold_2
                dataset    /splits/my_split_name_2/folds/fold_2/test_genome_idx
                dataset    /splits/my_split_name_2/folds/fold_2/train_genome_idx
                dataset    /splits/my_split_name_2/folds/fold_2/unique_risk_by_anti_kmer
                dataset    /splits/my_split_name_2/folds/fold_2/unique_risk_by_kmer
                dataset    /splits/my_split_name_2/folds/fold_2/unique_risks
 }
}
```
