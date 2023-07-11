#!/usr/bin/env python3

from gnn_tracking.models.track_condensation_networks import PreTrainedECGraphTCN
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from gnn_tracking_hpo.util.paths import add_scripts_path
from torch import nn

add_scripts_path()

from gnn_tracking_hpo.defaults import (
    legacy_config_compatibility,
    suggest_default_values,
)
from gnn_tracking_hpo.trainable import DefaultTrainable
from gnn_tracking_hpo.util.paths import get_config
from tune_ec import ECTrainable


# %%
def get_ec_config():
    project = "ec"
    hash = "a94b24d1"
    config = legacy_config_compatibility(get_config(project, hash))
    config["batch_size"] = 1
    config[
        "train_data_dir"
    ] = "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v2/part_1/"
    config[
        "val_data_dir"
    ] = "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v2/part_1/"
    config["n_graphs_train"] = 100
    config["n_graphs_val"] = 100
    config["_val_batch_size"] = 1
    config["m_use_intermediate_edge_embeddings"] = False
    config["m_hidden_dim"] = 30
    config["m_interaction_edge_dim"] = 30
    config["m_interaction_node_dim"] = 30
    config["m_L_ec"] = 5
    return config


def get_ec(config, freeze=True):
    trainable = ECTrainable(config)
    ec = trainable.trainer.model
    print(f"EC {len(list(ec.parameters()))=}")
    for param in ec.parameters():
        param.requires_grad = not freeze
    ec_params_trainable = [p for p in ec.parameters() if p.requires_grad]
    print(f"A {len(ec_params_trainable)=}")
    return ec


def get_hc_config():
    config = {
        "m_L_hc": 3,
        "m_hidden_dim": 30,
        "m_h_dim": 30,
        "m_e_dim": 30,
        "n_graphs_train": 100,
        "n_graphs_val": 100,
        "_val_batch_size": 1,
        "batch_size": 1,
        "m_mask_orphan_nodes": False,
        "test": False,
        "lw_edge": 100,
        "train_data_dir": "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v2/part_1",
        "val_data_dir": "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v2/part_9",
        "m_use_ec_embeddings_for_hc": True,
        "m_feed_edge_weights": False,
        "m_ec_threshold": 0.45,
    }
    suggest_default_values(config, ec="continued")
    return config


class PretrainedECTrainable(DefaultTrainable):
    def get_model(self) -> nn.Module:
        ec = get_ec(get_ec_config(), freeze=True)
        model = PreTrainedECGraphTCN(
            ec,
            node_indim=7,
            edge_indim=4,
            **subdict_with_prefix_stripped(self.tc, "m_"),
        )
        return model


trainable = PretrainedECTrainable(get_hc_config())
# trainable = ECTrainable(get_ec_config())
trainer = trainable.trainer
trainer.training_step(max_batches=300)
