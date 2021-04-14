# Modified from https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_gat.py
import argparse
import logging

import mlflow.keras
import numpy as np
from spektral.data.loaders import DisjointLoader
from spektral.layers import GATConv
from spektral.transforms import AdjToSpTensor, LayerPreprocess
from tensorflow.keras.callbacks import EarlyStopping

from GATModel import build_model
from ProteinDataset import ProteinDataset, mask_generator

EPOCHS_DEFAULT = 10000
PATIENCE_DEFAULT = 100
DATASET_DEFAULT = "simplified_data.pkl"

parser = argparse.ArgumentParser(description="Run a Graph Attention Model pipeline.")
parser.add_argument("dataset_pkl", help="Dataset (in PKL format) to use.")
parser.add_argument(
    "--name",
    help="Configuration name (e.g. 'contacts' or 'topology'). This will be tagged during the MLFlow run.",
    default=None,
)
parser.add_argument("--epochs", help="Number of epochs to run for.", default=10000)
parser.add_argument("--patience", help="Patience.", default=100)


def make_splits(graphs_pkl):
    # Generate train, val and test splits
    splits = dict.fromkeys(["train", "val", "test"])
    for split_name in splits:
        splits[split_name] = ProteinDataset(
            graphs_pkl,
            mask_func=mask_generator(split_name, 42, 0.70, 0.15),
            transforms=[LayerPreprocess(GATConv)],  # , AdjToSpTensor()
        )
    return splits


def make_loaders(splits):
    # Coalesce each split into a single graph
    loaders = dict.fromkeys(splits.keys())
    for split_name in loaders:
        loaders[split_name] = DisjointLoader(
            splits[split_name],
            node_level=True,
            batch_size=10,
            epochs=None,
            shuffle=True,
        )
    return loaders


def run_model(splits, loaders, epochs=EPOCHS_DEFAULT, patience=PATIENCE_DEFAULT):
    # Build and train model
    N = max((split.max_graph_size for split in splits.values()))
    F = splits["train"].n_node_features

    model = build_model(N, F)
    model.fit(
        loaders["train"].load(),
        steps_per_epoch=loaders["train"].steps_per_epoch,
        validation_data=loaders["val"].load(),
        validation_steps=loaders["val"].steps_per_epoch,
        epochs=epochs,
        callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
    )

    # Evaluate model
    print("Evaluating model.")
    eval_results = model.evaluate(
        loaders["test"].load(), steps=loaders["test"].steps_per_epoch
    )
    print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))


if __name__ == "__main__":
    mlflow.keras.autolog()
    args = parser.parse_args()

    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("patience", args.patience)
    if args.name is not None:
        mlflow.set_tag("name", args.name)

    splits = make_splits(args.dataset_pkl)
    loaders = make_loaders(splits)
    run_model(splits, loaders, epochs=args.epochs, patience=args.patience)
