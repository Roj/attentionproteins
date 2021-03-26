import logging

import tensorflow as tf
from spektral.layers import GATConv
from tensorflow.keras import metrics as keras_metrics
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

METRICS = [
    keras_metrics.TruePositives(name='tp'),
    keras_metrics.FalsePositives(name='fp'),
    keras_metrics.TrueNegatives(name='tn'),
    keras_metrics.FalseNegatives(name='fn'),
    keras_metrics.BinaryAccuracy(name='accuracy'),
    keras_metrics.Precision(name='precision'),
    keras_metrics.Recall(name='recall'),
    keras_metrics.AUC(name='auc'),
]

# Parameters
channels = 8  # Number of channels in each head of the first GAT layer
n_attn_heads = 8  # Number of attention heads in first GAT layer
dropout = 0.6  # Dropout rate for the features and adjacency matrix
l2_reg = 2.5e-4  # L2 regularization rate
learning_rate = 5e-3  # Learning rate
epochs = 20000  # Number of training epochs
patience = 100  # Patience for early stopping


def build_model(N, F):
    n_out = 1

    # Model definition
    x_in = Input(shape=(F,))
    a_in = Input((N,), sparse=True)
    e_in = Input(shape=(1,))

    do_1 = Dropout(dropout)(x_in)
    gc_1 = GATConv(
        channels,
        attn_heads=n_attn_heads,
        concat_heads=True,
        dropout_rate=dropout,
        activation="elu",
        kernel_regularizer=l2(l2_reg),
        attn_kernel_regularizer=l2(l2_reg),
        bias_regularizer=l2(l2_reg),
    )([do_1, a_in])
    do_2 = Dropout(dropout)(gc_1)
    gc_2 = GATConv(
        n_out,
        attn_heads=1,
        concat_heads=False,
        dropout_rate=dropout,
        activation="sigmoid",
        kernel_regularizer=l2(l2_reg),
        attn_kernel_regularizer=l2(l2_reg),
        bias_regularizer=l2(l2_reg),
    )([do_2, a_in])

    # Build model
    model = Model(inputs=[x_in, a_in, e_in], outputs=gc_2)
    optimizer = Adam(lr=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=BinaryCrossentropy(reduction="sum"),
        weighted_metrics=["acc"],
        metrics=METRICS,
    )
    logging.info("Model summary: %s", model.summary())
    return model
