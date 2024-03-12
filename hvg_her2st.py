import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")


# only need to run once to save hvg_matrix.npy
# filter expression matrices to only include HVGs shared across all datasets

def intersect_section_genes(adata_list):
    shared = set.intersection(*[set(adata.var_names) for adata in adata_list])
    return list(shared)


def her2_hvg_selection_and_pooling(adata_list, n_top_genes=1000):
    shared = intersect_section_genes(adata_list)

    hvg_bools = []

    for adata in adata_list:
        adata.var_names_make_unique()
        # Subset to shared genes
        adata = adata[:, shared]
        print(adata.shape)
        # Preprocess the data
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

        # save hvgs
        hvg = adata.var['highly_variable']
        hvg_bools.append(hvg)

    hvg_union = hvg_bools[0]
    hvg_intersection = hvg_bools[0]
    for i in range(1, len(hvg_bools)):
        print(sum(hvg_union), sum(hvg_bools[i]))
        hvg_union = hvg_union | hvg_bools[i]
        print(sum(hvg_intersection), sum(hvg_bools[i]))
        hvg_intersection = hvg_intersection & hvg_bools[i]

    print("Number of HVGs: ", hvg_union.sum())
    print("Number of HVGs (intersection): ", hvg_intersection.sum())

    with open('her2_hvgs_intersection.pickle', 'wb') as handle:
        pickle.dump(hvg_intersection, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('her2_hvgs_union.pickle', 'wb') as handle:
        pickle.dump(hvg_union, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Add all the HVGs

    gene_list_path = "D:\dataset\Her2st\data/her_hvg_cut_1000.npy"
    gene_list = list(np.load(gene_list_path, allow_pickle=True))

    hvg_union[gene_list] = True

    filtered_exp_mtxs = []
    for adata in adata_list:
        adata.var_names_make_unique()
        # Subset to shared genes
        adata = adata[:, shared]
        filtered_exp_mtxs.append(adata[:, gene_list].X.T.toarray())
    return filtered_exp_mtxs


names = os.listdir(".\dataset\Her2st\data/ST-cnts")
names.sort()
names = [i[:2] for i in names][1:33]

def her2_pool_gene_list(adata_list, n_top_genes=1000):
    gene_list_path = "D:\dataset\Her2st\data/her_hvg_cut_1000.npy"
    gene_list = list(np.load(gene_list_path, allow_pickle=True))


    filtered_exp_mtxs = []
    for adata in adata_list:
        adata.var_names_make_unique()
        filtered_exp_mtxs.append(adata[:, gene_list].X.T.toarray())
    return filtered_exp_mtxs

adata_list = [sc.AnnData(pd.read_csv(f".\dataset\Her2st\data/ST-cnts/{name}.tsv", sep='\t', index_col=0)) for name in
              names]
filtered_mtx = her2_pool_gene_list(adata_list)

import scprep as scp

preprocessed_mtx = []
for i, mtx in enumerate(filtered_mtx):

    normalized_expression = scp.normalize.library_size_normalize(mtx)
    log_transformed_expression = scp.transform.log(normalized_expression)
    preprocessed_mtx.append(log_transformed_expression)

    pathset = f"./data/preprocessed_expression_matrices/her2st/{names[i]}"
    if not os.path.exists(pathset):
        os.makedirs(pathset)
    np.save(f"./data/preprocessed_expression_matrices/her2st/{names[i]}/preprocessed_matrix.npy",
            log_transformed_expression)
    print(f"her_data_preprocessed_mtx[{i}]:", log_transformed_expression.shape)
