#!/usr/bin/env python

"""
@package
@brief
@author stfate
"""

import scipy as sp
import json
import adelheid.plot as pp


if __name__ == "__main__":
    bhtsne_result = json.load(open("./result/bhtsne_results.json"))
    bhtsne_samples = sp.array(list(bhtsne_result.keys())).astype(int)
    bhtsne_proctimes = sp.array(list(bhtsne_result.values()))

    tsnecuda_result = json.load(open("./result/tsnecuda_results.json"))
    tsnecuda_samples = sp.array(list(tsnecuda_result.keys())).astype(int)
    tsnecuda_proctimes = sp.array(list(tsnecuda_result.values()))

    umap_result = json.load(open("./result/umap_results.json"))
    umap_samples = sp.array(list(umap_result.keys())).astype(int)
    umap_proctimes = sp.array(list(umap_result.values()))

    fig,ax = pp.subplots()
    ax.semilogx(bhtsne_samples, bhtsne_proctimes, label="Barnes-Hut t-SNE")
    ax.semilogx(tsnecuda_samples, tsnecuda_proctimes, label="t-SNE-CUDA")
    ax.semilogx(umap_samples, umap_proctimes, label="UMAP")
    ax.set_xlabel("number of samples")
    ax.set_ylabel("processing time (sec)")
    ax.set_title("Processing time of t-SNE/t-SNE-CUDA/UMAP")
    ax.grid(True)
    ax.legend()
    # pp.show()
    pp.savefig("tsne-cuda-proctime.png")