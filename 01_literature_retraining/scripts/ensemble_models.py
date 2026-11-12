from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class MeanPredictionEnsemble:
    """Pickle-safe equal-weight regression ensemble."""

    def __init__(self, models: Sequence[Any], *, seeds: Sequence[int], model_name: str):
        self.models = list(models)
        self.seeds = [int(seed) for seed in seeds]
        self.model_name = str(model_name)

    def predict(self, X) -> np.ndarray:
        return np.mean(
            np.vstack([np.asarray(model.predict(X), dtype=float) for model in self.models]),
            axis=0,
        )
