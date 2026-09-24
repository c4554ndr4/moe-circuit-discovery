"""Stream exact route-weighted neuron means; each labeled example has equal weight."""

import numpy as np


class ActivationStats:
    def __init__(self, dims):
        self.dims = dims
        self.sums = np.zeros((2, *dims), dtype=np.float64)
        self.positive_example_hits = np.zeros(dims, dtype=np.int64)
        self.counts = [0, 0]

    def begin(self, start, stop):
        if stop <= start:
            raise ValueError("Empty response token window")
        self.start, self.stop = start, stop
        self.current = np.zeros(self.dims, dtype=np.float64)
        self.touched = np.zeros(self.dims[:2], dtype=bool)
        self.layers = set()

    def observe(self, layer, expert, positions, gated, weights):
        self.layers.add(layer)
        mask = (positions >= self.start) & (positions < self.stop)
        if not mask.any():
            return
        product = (gated[mask].float() * weights[mask].float()).sum(dim=0).detach().cpu().numpy()
        if not np.isfinite(product).all():
            raise ValueError("Non-finite expert activations")
        self.current[layer, expert] += product / (self.stop - self.start)
        self.touched[layer, expert] = True

    def finish(self, label):
        if self.layers != set(range(self.dims[0])):
            raise ValueError("Missing expert activations in one or more layers")
        self.sums[label] += self.current
        self.counts[label] += 1
        if label == 1:
            self.positive_example_hits += self.touched[:, :, None]

    def rank(self, top_k, min_positive_examples=1, mode="absolute"):
        if min(self.counts) == 0:
            raise ValueError("Both labels need activation measurements")
        positive = self.sums[1] / self.counts[1]
        negative = self.sums[0] / self.counts[0]
        delta = positive - negative
        score = np.abs(delta) if mode == "absolute" else delta
        eligible = (self.positive_example_hits >= min_positive_examples) & (score > 0)
        candidates = np.flatnonzero(eligible)
        order = candidates[np.argsort(-score.ravel()[candidates], kind="stable")[:top_k]]
        if not len(order):
            raise ValueError("No eligible neurons; inspect labels, data, and --min-positive-examples")
        rows = []
        for flat in order:
            layer, expert, neuron = (int(i) for i in np.unravel_index(flat, self.dims))
            key = (layer, expert, neuron)
            rows.append({"layer": layer, "expert": expert, "neuron": neuron,
                         "score": float(score[key]), "signed_delta": float(delta[key]),
                         "positive_mean": float(positive[key]), "negative_mean": float(negative[key]),
                         "positive_examples_routing_expert": int(self.positive_example_hits[key])})
        return rows


def matched_random(targets, dims, seed):
    """Match counts within each layer/expert, excluding the candidate neurons."""
    rng = np.random.default_rng(seed)
    groups = {}
    for row in targets:
        groups.setdefault((row["layer"], row["expert"]), set()).add(row["neuron"])
    result = []
    for (layer, expert), excluded in sorted(groups.items()):
        available = np.array([i for i in range(dims[2]) if i not in excluded])
        if len(available) < len(excluded):
            raise ValueError("Too many targets in an expert to construct a disjoint size-matched control")
        for neuron in rng.choice(available, size=len(excluded), replace=False):
            result.append({"layer": layer, "expert": expert, "neuron": int(neuron)})
    return result
