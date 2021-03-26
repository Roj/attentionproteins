# %%
# Modified from https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_gat.py
import logging

import numpy as np
from spektral.data.loaders import DisjointLoader
from spektral.layers import GATConv
from spektral.transforms import AdjToSpTensor, LayerPreprocess
from tensorflow.keras.callbacks import EarlyStopping

from GATModel import build_model
from ProteinDataset import ProteinDataset, mask_generator

GRAPHS_PKL = "simplified_data.pkl"
EPOCHS = 10000
PATIENCE = 100

# Generate train, val and test splits
splits = dict.fromkeys(["train", "val", "test"])
for split_name in splits:
    splits[split_name] = ProteinDataset(
        GRAPHS_PKL,
        mask_func=mask_generator(split_name, 42, 0.70, 0.15),
        transforms=[LayerPreprocess(GATConv)],  # , AdjToSpTensor()
    )

# Coalesce each split into a single graph
loaders = dict.fromkeys(splits.keys())
for split_name in loaders:
    loaders[split_name] = DisjointLoader(
        splits[split_name], node_level=True, batch_size=10, epochs=None, shuffle=True,
    )

# Build and train model
N = max((split.max_graph_size for split in splits.values()))
F = splits["train"].n_node_features

model = build_model(N, F)
model.fit(
    loaders["train"].load(),
    steps_per_epoch=loaders["train"].steps_per_epoch,
    validation_data=loaders["val"].load(),
    validation_steps=loaders["val"].steps_per_epoch,
    epochs=EPOCHS,
    callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True)],
)

# Evaluate model
print("Evaluating model.")
loader_te = SingleLoader(dataset, sample_weights=weights_te)
eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))