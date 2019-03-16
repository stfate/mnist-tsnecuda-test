#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@package test_mnist_tsnecuda.py
@brief
@author stfate
"""

import scipy as sp
from sklearn.datasets import fetch_mldata
import json
from tqdm import tqdm
from tsnecuda import TSNE
import umi.plot as pp

import time


if __name__ == "__main__":
    COLORS = ["#000000", "#ffff33", "#66ff66", "#00ccff", "#6633ff", "#cc3399", "#ff3300", "#0066cc", "#9933cc", "#006633"]
    mnist = fetch_mldata("MNIST original", data_home="./")
    n_obs_all = len(mnist["data"])
    n_obs_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 10000, 20000, 30000, 40000, 50000, 60000, n_obs_all]

    result = {}
    for n_obs in tqdm(n_obs_list):
        mnist_data = mnist["data"][:n_obs]
        mnist_labels = mnist["target"][:n_obs].astype(int)

        st = time.clock()

        model = TSNE(n_components=2, perplexity=30.0, theta=0.5, n_iter=1000)
        predicted = model.fit_transform(mnist_data)

        ed = time.clock()
        result[n_obs] = ed-st

        xmin = predicted[:,0].min()
        xmax = predicted[:,0].max()
        ymin = predicted[:,1].min()
        ymax = predicted[:,1].max()

        output_fn = f"mnist_tsnecuda_obs{n_obs}.png"
        fig,ax = pp.subplots(figsize=(16,12))
        unique_labels = sp.unique(mnist_labels)
        for _label in unique_labels:
            inds = sp.where(mnist_labels == _label)[0]
            ax.scatter(predicted[inds,0], predicted[inds,1], c=COLORS[_label], label=_label)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("component 0")
        ax.set_ylabel("component 1")
        ax.set_title("MNIST t-SNE-CUDA visualization")
        ax.legend()
        pp.savefig(output_fn)

    json.dump(result, open("tsnecuda_results.json", "w"), ensure_ascii=False, indent=2, sort_keys=True)
