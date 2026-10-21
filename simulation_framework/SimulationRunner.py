"""SimulationConfig generator and parallel simulation runner"""

from dataclasses import replace
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from tqdm.auto import tqdm

from .data_classes import AggregatedResults, SimulationConfig, SimulationResult
from .DGP import generate_data

DEFAULT_COVARIATE_SCHEMA = {
    "normal": 0.50,
    "binary": 0.30,
    "ordinal": 0.10,
    "lognormal": 0.10
}

DEFAULT_COVARIATE_PARAMS = {
    "binary_prevalence": (0.2, 0.8),
    "ordinal_levels": (3, 5),
    "ordinal_concentration": 2.0,
    "lognormal_sigma": (0.5, 1.0)
}

DEFAULT_LINEAR_OUTCOME = {
    "target_r_squared": 0.3
}

DEFAULT_SURVIVAL_OUTCOME = {
    "target_event_rate": 0.3,
    "max_followup": 120.0,
    "hazard_shape": 1.0,
    "p_loss_to_followup": 0.10,
    "time_resolution": 1.0
}

def create_config(
    name = "realistic_clinical",

    n_samples = 1000,
    n_features = 200,

    n_strong = 3,
    n_moderate = 5,
    n_weak = 12,
    strong_range = (1.0, 1.0),
    moderate_range = (0.5, 0.5),
    weak_range = (0.25, 0.25),
    align_signs = True,

    n_clusters = 30,
    cluster_size = (2, 7),
    isolated_feature_fraction = 0.2,
    within_cluster_correlation = (0.3, 0.8),
    positive_feature_rate = 0.8,

    copula_family = "gaussian",
    copula_df = None,

    covariate_schema = DEFAULT_COVARIATE_SCHEMA,
    covariate_params = DEFAULT_COVARIATE_PARAMS,

    outcome_type = "survival",
    outcome_params = None,

    random_state = 42
) -> SimulationConfig:
    """
    Coefficient magnitudes are uniform within each tier's range. align_signs
    uses cluster signs for clustered signals and random signs for isolated ones.
    within_cluster_correlation specifies target Spearman magnitudes.
    positive_feature_rate is a per-feature sign probability; the positive-pair
    probability is p**2 + (1-p)**2. copula_df controls t-copula tail dependence.
    """

    if outcome_params is None:
        if outcome_type == "survival":
            outcome_params = dict(DEFAULT_SURVIVAL_OUTCOME)
        elif outcome_type == "linear":
            outcome_params = dict(DEFAULT_LINEAR_OUTCOME)
        else:
            raise ValueError("Unknown outcome_type.")

    corr_params = {
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "isolated_feature_fraction": isolated_feature_fraction,
        "within_cluster_correlation": within_cluster_correlation,
        "positive_feature_rate": positive_feature_rate
    }
    coef_params = {
        "n_strong": n_strong,
        "n_moderate": n_moderate,
        "n_weak": n_weak,
        "strong_range": tuple(strong_range),
        "moderate_range": tuple(moderate_range),
        "weak_range": tuple(weak_range),
        "align_signs": align_signs
    }
    copula_params = {
        "copula_family": copula_family,
        "copula_df": copula_df
    }

    return SimulationConfig(
        name=name,
        n_samples=n_samples,
        n_features=n_features,
        corr_params=corr_params,
        coef_params=coef_params,
        copula_params=copula_params,
        outcome_type=outcome_type,
        outcome_params=outcome_params,
        covariate_schema=covariate_schema,
        covariate_params=covariate_params,
        random_state=random_state
    )


class TqdmParallel(Parallel):
    """tqdm progress bar"""
    def __init__(self, total, desc, **kwargs):
        self._pbar = tqdm(total=total, desc=desc, leave=False)
        super().__init__(**kwargs)
        
    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        finally:
            self._pbar.close()
    
    def print_progress(self):
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


class SimulationRunner:
    """Parallel simulation runner"""
    def __init__(
        self,
        methods,
        n_repeats,
        n_jobs = -1,
        save_path = None,
        verbose = True
    ):

        self.methods = methods
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.save_path = Path(save_path or "raw_results.csv")

        self.shard_dir = self.save_path.parent / (self.save_path.stem + "_shards")
        self.shard_dir.mkdir(parents=True, exist_ok=True)

        self.diagnostics_dir = (
            self.save_path.parent / (self.save_path.stem + "_diagnostics")
        )
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    def add_method(self, func):
        self.methods.append(func)

    @staticmethod
    def _safe(s):
        return "".join(c if c.isalnum() or c in "._-=" else "_" for c in s)

    def _task_id(self, config, func, repeat_idx):
        func_name = getattr(func, "__name__", func.__class__.__name__)
        return self._safe(
            f"{config.name}__base{config.random_state}"
            f"__{func_name}__rep{repeat_idx}"
        )

    def _shard_path(self, task_id):
        return self.shard_dir / f"{task_id}.jsonl"

    def _write_json(self, path, data, lines=False):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            if lines:
                for row in data:
                    f.write(json.dumps(row) + "\n")
            else:
                json.dump(data, f, default=self._json_default, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _write_jsonl(self, path, rows):
        if rows:
            self._write_json(path, rows, lines=True)

    def _diagnostics_path(self, config, repeat_idx):
        name = self._safe(
            f"{config.name}__base{config.random_state}__rep{repeat_idx}"
        )
        return self.diagnostics_dir / f"{name}.json"

    @staticmethod
    def _json_default(obj):
        """Convert numpy types to native types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)

    def _save_diagnostics(self, config, repeat_idx, diagnostics):
        self._write_json(self._diagnostics_path(config, repeat_idx), diagnostics)

    @staticmethod
    def _derive_seed(base_random_state, repeat_idx):
        """Derive a per-repeat seed from (base, repeat_idx)"""
        ss = np.random.SeedSequence(
            entropy=int(base_random_state),
            spawn_key=(int(repeat_idx),)
        )
        return int(ss.generate_state(1)[0])

    @staticmethod
    def _result_row(config, true_features, method, selected, elapsed_time):
        return SimulationResult(
            config_name=config.name,
            outcome_type=config.outcome_type,
            n_samples=config.n_samples,
            n_features=config.n_features,
            true_features=true_features,
            method=method,
            selected=selected,
            elapsed_time=elapsed_time
        ).to_dict()

    def _run_single_repeat(self, config, repeat_idx):
        """Generate data once and run methods"""
        pending = []
        for func in self.methods:
            out_path = self._shard_path(self._task_id(config, func, repeat_idx))
            if not out_path.exists():
                pending.append((func, out_path))

        diag_path = self._diagnostics_path(config, repeat_idx)
        need_diagnostics = not diag_path.exists()

        if not pending and not need_diagnostics:
            return

        derived_seed = self._derive_seed(config.random_state, repeat_idx)
        config_copy = replace(config, random_state=derived_seed)

        try:
            X, y, true_features, diagnostics = generate_data(config_copy)
        except Exception as e:
            if self.verbose:
                warnings.warn(
                    f"Data generation failed for config={config.name!r}, "
                    f"repeat={repeat_idx}, derived_seed={derived_seed}: {e}"
                )
            for func, out_path in pending:
                row = self._result_row(config, [], "__DATA_GEN_FAILED__", [], 0.0)
                self._write_jsonl(out_path, [row])
            return

        if need_diagnostics:
            try:
                self._save_diagnostics(config, repeat_idx, diagnostics)
            except Exception as e:
                if self.verbose:
                    warnings.warn(
                        f"Failed to save diagnostics for "
                        f"config={config.name!r}, repeat={repeat_idx}: {e}"
                    )

        if not pending:
            return

        for func, out_path in pending:
            start_time = time.time()
            try:
                selection_output = func(X, y)  # {method_name: [features]}
            except Exception as e:
                if self.verbose:
                    warnings.warn(
                        f"Feature selection failed for "
                        f"{getattr(func, '__name__', repr(func))} "
                        f"on config={config.name!r}, repeat={repeat_idx}: {e}"
                    )
                selection_output = {}
            elapsed = time.time() - start_time

            rows = [
                self._result_row(
                    config_copy, true_features, method_name,
                    selection_output.get(method_name, []), elapsed
                )
                for method_name in (list(selection_output.keys()) or ["__FAILED__"])
            ]
            self._write_jsonl(out_path, rows)

    def run(self, configs):
        """Run all simulations and return an AggregatedResults wrapper"""

        tasks = [
            (config, repeat_idx)
            for config in configs
            for repeat_idx in range(self.n_repeats)
        ]
        total_tasks = len(tasks)

        if self.verbose:
            print(f"Running {total_tasks} (config, repeat) tasks...")
            print(f"  Configs: {len(configs)}")
            print(f"  Methods per task: {len(self.methods)}")
            print(f"  Repeats per config: {self.n_repeats}")

        if self.verbose:
            parallel = TqdmParallel(
                total=total_tasks, desc="Simulations", n_jobs=self.n_jobs
            )
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
        parallel(
            delayed(self._run_single_repeat)(config, repeat_idx)
            for config, repeat_idx in tasks
        )

        frames = [
            pd.read_json(path, lines=True)
            for path in sorted(self.shard_dir.glob("*.jsonl"))
        ]
        results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        results.to_csv(self.save_path, index=False)

        return AggregatedResults(results)
