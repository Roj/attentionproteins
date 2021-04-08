# Modified from https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_gat.py
import logging

import mlflow.keras
import numpy as np
from spektral.data.loaders import DisjointLoader
from spektral.layers import GATConv
from spektral.transforms import AdjToSpTensor, LayerPreprocess
from tensorflow.keras.callbacks import EarlyStopping

from GATModel import build_model
from ProteinDataset import ProteinDataset, mask_generator

mlflow.keras.autolog()

GRAPHS_PKL = "simplified_data.pkl"
EPOCHS = 100
PATIENCE = 10

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
eval_results = model.evaluate(
    loaders["test"].load(), steps=loaders["test"].steps_per_epoch
)
print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))
