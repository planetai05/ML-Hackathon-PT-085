"""Export inference artifacts (issue #19).

python -m models.export --model lgbm_tuned

artifacts/
  model.joblib         fitted classifier (scaling, where used, is inside the sklearn Pipeline)
  model_meta.json      model name, source run, params, training data, metrics, versions
  feature_config.json  feature list + candidate parameters (written by features.build_features)
  hmm_config.json      tuned HMM/Viterbi parameters (written by evaluation.route_eval)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import sklearn

from hmm.model import HMMConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    os.makedirs("artifacts", exist_ok=True)
    src = f"models/saved/{a.model}.joblib"
    shutil.copyfile(src, "artifacts/model.joblib")
    run = json.load(open(f"results/runs/{a.model}.json"))
    if not os.path.exists("artifacts/hmm_config.json"):
        json.dump(HMMConfig().to_dict(), open("artifacts/hmm_config.json", "w"), indent=1)
    import lightgbm
    meta = dict(model_name=a.model, model_file="model.joblib", family=run["family"], params=run["params"], seed=run["seed"],
                trained_on="all 80 train trips (aligned points)", cv_mean=run["cv_mean"], student_test=run["test"],
                artifact_version=datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M"),
                versions=dict(python=sys.version.split()[0], sklearn=sklearn.__version__, lightgbm=lightgbm.__version__))
    json.dump(meta, open("artifacts/model_meta.json", "w"), indent=1, default=float)
    print("exported", a.model, "->", os.path.getsize("artifacts/model.joblib") // 1024, "KB")


if __name__ == "__main__":
    main()
