from dataclasses import dataclass
import pandas as pd


@dataclass
class SimulationConfig:
    """Configuration for a single simulation scenario"""
    name: str
    n_samples: int
    n_features: int
    corr_params: dict
    coef_params: dict
    copula_params: dict
    outcome_type: str
    outcome_params: dict
    covariate_schema: dict
    covariate_params: dict
    random_state: int


@dataclass
class SimulationResult:
    """Results from a single simulation run"""
    config_name: str
    outcome_type: str
    n_samples: int
    n_features: int
    true_features: list

    method: str
    selected: list
    n_selected: int = 0
    is_empty: bool = False
    elapsed_time: float = 0.0

    TP: int = 0
    FP: int = 0
    TN: int = 0
    FN: int = 0

    def __post_init__(self):
        true_set = set(self.true_features)
        selected = set(self.selected)

        self.TP = len(true_set & selected)
        self.FP = len(selected - true_set)
        self.FN = len(true_set - selected)
        self.TN = self.n_features - len(true_set | selected)
        self.n_selected = len(self.selected)
        self.is_empty = (self.n_selected == 0)

    @property
    def sensitivity(self):
        total = self.TP + self.FN
        return self.TP / total if total else 0.0

    @property
    def specificity(self):
        total = self.TN + self.FP
        return self.TN / total if total else 0.0

    @property
    def precision(self):
        if self.is_empty:
            return 0.0
        return self.TP / (self.TP + self.FP)

    @property
    def fdr(self):
        if self.is_empty:
            return 0.0
        return 1.0 - self.precision

    @property
    def f1_score(self):
        p, s = self.precision, self.sensitivity
        return 2 * (p * s) / (p + s) if p + s else 0.0


    def to_dict(self):
        return {
            "scenario": self.config_name,
            "outcome_type": self.outcome_type,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_true_features": len(self.true_features),
            "method": self.method,
            "selected_set": self.selected,
            "n_selected": self.n_selected,
            "is_empty": self.is_empty,
            "elapsed_time": self.elapsed_time,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
            "fdr": self.fdr,
            "f1_score": self.f1_score
        }


@dataclass
class AggregatedResults:
    """Aggregated results across multiple simulation runs"""
    results: pd.DataFrame

    def summary(self, groupby=["scenario", "method"]):
        metrics = [
            "sensitivity", "specificity",
            "precision", "fdr", "f1_score",
            "n_selected", "elapsed_time"
        ]

        grouped = self.results.groupby(groupby)
        summary = grouped[metrics].agg(["mean", "std", "median"])
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()

        empty_rate = (
            grouped["is_empty"]
            .mean()
            .rename("empty_selection_rate")
            .reset_index()
        )
        return summary.merge(empty_rate, on=groupby, how="left")
