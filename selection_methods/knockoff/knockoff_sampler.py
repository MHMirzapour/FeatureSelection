"""SCIP knockoff sampler for mixed data types"""

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd


@dataclass
class SeqKnockoffSampler:
    l1_ratio: float = 0.5
    n_cv: int = 5
    n_alphas: int = 50
    alpha_min_ratio: float | None = None
    n_jobs: int = 1
    random_state: int = 42
    rscript: str = "Rscript"

    def sample(self, X, categorical_features = []):
        is_df = isinstance(X, pd.DataFrame)
        df = pd.DataFrame(X).copy()
        p = df.shape[1]

        categorical = set()
        for feature in categorical_features:
            if isinstance(feature, (int, np.integer)):
                categorical.add(feature)
            elif is_df and isinstance(feature, str):
                categorical.add(df.columns.get_loc(feature))

        encoded, n_levels, categories = [], [], {}
        for j in range(p):
            column = df.iloc[:, j]
            if j in categorical:
                cat = pd.Categorical(column).remove_unused_categories()
                encoded.append(cat.codes.astype(int) + 1)
                categories[j] = cat.categories
                n_levels.append(len(cat.categories))
            else:
                encoded.append(column.to_numpy())
                n_levels.append(0)

        executable = shutil.which(self.rscript)
        script = Path(__file__).with_name("seqknockoff.R")

        with tempfile.TemporaryDirectory(prefix="seqknockoff_") as directory:
            input_path = Path(directory) / "X.csv"
            output_path = Path(directory) / "Xk.csv"
            pd.DataFrame(np.column_stack(encoded), columns=[f"V{i+1}" for i in range(p)]).to_csv(
                input_path, index=False
            )
            command = [
                executable, "--vanilla", str(script), str(input_path), str(output_path),
                ",".join(map(str, n_levels)), str(self.random_state), str(self.l1_ratio), str(self.n_cv),
                str(self.n_alphas), str(int(self.n_jobs)),
                "" if self.alpha_min_ratio is None else str(self.alpha_min_ratio)
            ]
            process = subprocess.run(command, capture_output=True, text=True)
            values = pd.read_csv(output_path).to_numpy()

        columns = []
        for j in range(p):
            values_j = values[:, j]
            if j in categories:
                codes = np.rint(values_j).astype(int)
                columns.append(pd.Series(categories[j].take(codes - 1), index=df.index).astype(df.iloc[:, j].dtype))
            else:
                columns.append(pd.Series(values_j, index=df.index))
        result = pd.concat(columns, axis=1)
        result.columns = df.columns
        return result if is_df else result.to_numpy()
