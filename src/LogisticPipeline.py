import numpy as np
from tensorflow.keras import metrics as keras_metrics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from ProteinDataset import ProteinDataset, mask_generator

EPOCHS_DEFAULT = 100
PATIENCE_DEFAULT = 10
METRICS = [
    keras_metrics.TruePositives(name="tp"),
    keras_metrics.FalsePositives(name="fp"),
    keras_metrics.TrueNegatives(name="tn"),
    keras_metrics.FalseNegatives(name="fn"),
    keras_metrics.BinaryAccuracy(name="accuracy"),
    keras_metrics.Precision(name="precision"),
    keras_metrics.Recall(name="recall"),
    keras_metrics.AUC(name="auc"),
]

parser = argparse.ArgumentParser(description="Run a Logistic Regression pipeline.")
parser.add_argument("dataset_pkl", help="Dataset (in PKL format) to use.")
parser.add_argument(
    "--name",
    help="Configuration name (e.g. 'contacts' or 'topology'). This will be tagged during the MLFlow run.",
    default=None,
)
parser.add_argument("--epochs", help="Number of epochs to run for.", default=10000)
parser.add_argument("--patience", help="Patience.", default=100)


split_names = ["train", "val", "test"]


def make_splits(graphs_pkl):
    # Generate train, val and test splits
    splits = dict.fromkeys(split_names)
    for split_name in splits:
        splits[split_name] = ProteinDataset(
            graphs_pkl, mask_func=mask_generator(split_name, 42, 0.70, 0.15)
        )
    return splits


def get_attr_from_splits(splits, attr_name):
    return {
        split_name: np.concatenate(
            [graph.__getattr__(attr_name) for graph in split.read()]
        )
        for split_name, split in splits.items()
    }


def unstack_splits(splits, names=split_names):
    return [splits[name] for name in names]


def run_model(splits, epochs=EPOCHS_DEFAULT, patience=PATIENCE_DEFAULT):
    features = get_attr_from_splits(splits, "x")
    targets = get_attr_from_splits(splits, "y")

    X_train, X_val, X_test = unstack_splits(features)
    Y_train, Y_val, Y_test = unstack_splits(targets)

    model = Sequential()
    model.add(Dense(1, activation="sigmoid", input_dim=X_train.shape[1]))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=METRICS)
    model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        validation_data=(X_val, Y_val),
        callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
    )


if __name__ == "__main__":
    mlflow.keras.autolog()

    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("patience", args.patience)
    mlflow.set_tag("model", "LR")
    if args.name is not None:
        mlflow.set_tag("name", args.name)

    splits = make_splits("simplified_data.pkl")
    run_model(splits)
