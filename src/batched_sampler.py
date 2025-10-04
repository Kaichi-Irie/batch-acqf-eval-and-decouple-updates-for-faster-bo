import time
from typing import Literal

import numpy as np
import optuna._gp.acqf as acqf_module
from optuna.samplers import GPSampler

from src import (
    batched_acqf_eval_optim_mixed,
    simplified_optim_mixed,
    stacking_optim_mixed,
)

SAMPLERMODE = Literal["stacking", "batched_acqf_eval", "original"]


class BatchedSampler(GPSampler):
    def __init__(self, mode: SAMPLERMODE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode: SAMPLERMODE = mode
        self.average_nits = []
        self.elapsed_acqf_opt = 0.0

    def _optimize_acqf(
        self, acqf: acqf_module.BaseAcquisitionFunc, best_params: np.ndarray | None
    ) -> np.ndarray:
        assert best_params is None or len(best_params.shape) == 2
        start = time.time()
        if self.mode == "stacking":
            # batched_size is set as n_local_search (=10) inside stacking_optim_mixed
            normalized_params, _acqf_val = stacking_optim_mixed.optimize_acqf_mixed(
                acqf,
                warmstart_normalized_params_array=best_params,
                n_preliminary_samples=self._n_preliminary_samples,
                n_local_search=self._n_local_search,
                tol=self._tol,
                rng=self._rng.rng,
            )
            elapsed = time.time() - start
            self.elapsed_acqf_opt += elapsed
            return normalized_params
            # raise ValueError("Stacking mode is not implemented.")
        if self.mode == "batched_acqf_eval":
            normalized_params, _acqf_val = (
                batched_acqf_eval_optim_mixed.optimize_acqf_mixed(
                    acqf,
                    warmstart_normalized_params_array=best_params,
                    n_preliminary_samples=self._n_preliminary_samples,
                    n_local_search=self._n_local_search,
                    tol=self._tol,
                    rng=self._rng.rng,
                )
            )
            elapsed = time.time() - start
            self.elapsed_acqf_opt += elapsed
            return normalized_params
        if self.mode == "original":
            param, _acqf_val = simplified_optim_mixed.optimize_acqf_mixed(
                acqf,
                warmstart_normalized_params_array=best_params,
                n_preliminary_samples=self._n_preliminary_samples,
                n_local_search=self._n_local_search,
                tol=self._tol,
                rng=self._rng.rng,
            )
            elapsed = time.time() - start
            self.elapsed_acqf_opt += elapsed
            return param
        else:
            raise NotImplementedError(f"mode {self.mode} is not implemented.")
