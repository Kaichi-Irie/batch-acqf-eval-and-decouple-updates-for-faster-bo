import time
from typing import Literal

import numpy as np

import optuna._gp.acqf as acqf_module
from optuna.samplers import GPSampler

from . import (
    cbe_optim_mixed,
    dbe_optim_mixed,
    seqopt_optim_mixed,
)

SAMPLERMODE = Literal[
    "coupled_batch_evaluation", "decoupled_batch_evaluation", "original"
]


class BatchedSampler(GPSampler):
    def __init__(self, mode: SAMPLERMODE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode: SAMPLERMODE = mode
        self.nit_stats_list = []
        self.elapsed_acqf_opt = 0.0

    def _optimize_acqf(
        self, acqf: acqf_module.BaseAcquisitionFunc, best_params: np.ndarray | None
    ) -> np.ndarray:
        assert best_params is None or len(best_params.shape) == 2
        start = time.time()
        if self.mode == "coupled_batch_evaluation":
            (
                normalized_params,
                _acqf_val,
                nit_stats,
            ) = cbe_optim_mixed.optimize_acqf_mixed(
                acqf,
                warmstart_normalized_params_array=best_params,
                n_preliminary_samples=self._n_preliminary_samples,
                n_local_search=self._n_local_search,
                tol=self._tol,
                rng=self._rng.rng,
            )
            self.nit_stats_list.append(nit_stats)
            elapsed = time.time() - start
            self.elapsed_acqf_opt += elapsed
            return normalized_params
        if self.mode == "decoupled_batch_evaluation":
            (
                normalized_params,
                _acqf_val,
                nit_stats,
            ) = dbe_optim_mixed.optimize_acqf_mixed(
                acqf,
                warmstart_normalized_params_array=best_params,
                n_preliminary_samples=self._n_preliminary_samples,
                n_local_search=self._n_local_search,
                tol=self._tol,
                rng=self._rng.rng,
            )
            self.nit_stats_list.append(nit_stats)
            elapsed = time.time() - start
            self.elapsed_acqf_opt += elapsed
            return normalized_params
        if self.mode == "original":
            param, _acqf_val, nit_stats = seqopt_optim_mixed.optimize_acqf_mixed(
                acqf,
                warmstart_normalized_params_array=best_params,
                n_preliminary_samples=self._n_preliminary_samples,
                n_local_search=self._n_local_search,
                tol=self._tol,
                rng=self._rng.rng,
            )
            self.nit_stats_list.append(nit_stats)
            elapsed = time.time() - start
            self.elapsed_acqf_opt += elapsed
            return param
        else:
            raise NotImplementedError(f"mode {self.mode} is not implemented.")
