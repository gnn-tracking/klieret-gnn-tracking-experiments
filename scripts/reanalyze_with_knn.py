import json
import logging
import typing
from pathlib import Path

import numpy as np
from gnn_tracking.analysis.graphs import get_largest_segment_fracs
from gnn_tracking_hpo.trainable import MLTrainable
from torch_cluster import knn_graph
from torch_geometric.data import Data
from tqdm import tqdm


def summarize_ls(ls):
    return {
        "frac_segment50": np.mean(ls > 0.5),
        "frac_segment75": np.mean(ls > 0.75),
        "frac_segment100": np.mean(ls == 1.0),
    }


def construct_graph_knn(mo: dict[str, typing.Any], k, max_edges=5e7):
    edge_index = knn_graph(mo["x"], k)
    if edge_index.shape[1] > max_edges:
        raise RuntimeError(f"Too many edges: {edge_index.shape[1]}")
    y: Tensor = (  # type: ignore
        mo["particle_id"][edge_index[0, :]] == mo["particle_id"][edge_index[1, :]]
    )
    data = Data(x=mo["x"], edge_index=edge_index, y=y)
    data.pt = mo["particle_id"]
    data.particle_id = mo["particle_id"]
    return data


def get_max_knn_stats(project, hash, epoch=-1):
    trainable = MLTrainable.reinstate(project, hash, epoch=epoch)
    pc = trainable.trainer.val_loader.dataset[0]
    mo = trainable.trainer.evaluate_model(pc)
    data = construct_graph_knn(mo, 99, max_edges=1e8)
    return summarize_ls(get_largest_segment_fracs(data)) | {
        "n_edges": data.num_edges,
        "k": 99,
        "hash": hash,
        "epoch": epoch,
        "project": project,
    }


if __name__ == "__main__":
    dirs = [p for p in Path("/home/kl5675/ray_results/gc").iterdir() if p.is_dir()]
    broken = []
    results = []
    logging.disable(logging.CRITICAL)
    for d in tqdm(dirs):
        hash = d.name.split("_")[1]
        available_epochs = [
            int(p.name.split("checkpoint_")[1])
            for p in d.iterdir()
            if p.name.startswith("checkpoint_")
        ]
        for epoch in available_epochs:
            try:
                result = get_max_knn_stats("gc", hash, epoch=epoch)
            except Exception as e:
                print(e)
                broken.append(
                    {
                        "hash": hash,
                        "epoch": epoch,
                        "error": e,
                    }
                )
                with open("broken.json", "w") as f:
                    json.dump(results, f)
                continue
            results.append(result)
            with open("results.json", "w") as f:
                json.dump(results, f)
