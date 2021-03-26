import pickle as pkl

import numpy as np
import spektral.data


class ProteinDataset(spektral.data.Dataset):
    """Protein dataset from a pickled file"""

    def __init__(self, pkl_file, mask_func=None, **kwargs):
        self.pkl_file = pkl_file
        self.graphs = []
        self.mask_func = mask_func
        super().__init__(**kwargs)

    def download(self):
        """Load data from pickle file"""
        with open(self.pkl_file, "rb") as f:
            self.data = pkl.load(f)

    def read(self):
        if len(self.graphs) > 0:
            return self.graphs
        self.max_graph_size = next(iter(self.data.values()))["features"].shape[0]
        for group in self.data.values():
            self.max_graph_size = max(self.max_graph_size, group["features"].shape[0])
            self.graphs.append(
                spektral.data.Graph(
                    x=group["features"],
                    a=group["adj"],
                    y=group["target"].reshape((-1, 1)),
                )
            )
        if self.mask_func is None:
            return self.graphs
        mask = self.mask_func(len(self.graphs))
        return [graph for graph, selected in zip(self.graphs, mask) if selected]


def mask_generator(mask_type, seed, train_perc, val_perc):
    """Make a mask generator for train/val/test according to seed and
    split percentages. A function is returned that generates
    a mask given a size."""
    valid_types = ["train", "val", "test"]
    if mask_type not in valid_types:
        raise ValueError(f"Mask '{mask_type}' not understood")
    mask_id = valid_types.index(mask_type)
    rng = np.random.default_rng(seed)

    def build_mask(N):
        """Generate a mask using configured losure"""
        mask = rng.choice(
            [0, 1, 2],
            size=N,
            replace=True,
            p=[train_perc, val_perc, 1 - val_perc - train_perc],
        )
        return mask == mask_id

    return build_mask
