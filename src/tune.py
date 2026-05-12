from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None

from feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN, create_features
from seed_everything import seed_everything
from train import artifact_stem, evaluate_predictions
from utils import MODEL_DIR, PROCESSED_DATA_PATH, PROJECT_ROOT, REPORT_DIR, ensure_directories, load_csv

EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
TUNABLE_MODELS = ["random_forest", "xgboost", "lightgbm"]


def target_prefix(target: str) -> str:
    return "risk_" if target == "at_risk_user" else ""


def suggest_model(trial: optuna.Trial, model_name: str, random_state: int = 42):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 24),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("Install xgboost to tune xgboost.")
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
            eval_metric="logloss",
            random_state=random_state,
        )
    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("Install lightgbm to tune lightgbm.")
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
            num_leaves=trial.suggest_int("num_leaves", 16, 128),
            max_depth=trial.suggest_int("max_depth", 3, 16),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
            random_state=random_state,
            verbose=-1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def load_training_data(seed: int = 42, target: str = TARGET_COLUMN, data_path: Path = PROCESSED_DATA_PATH):
    data = load_csv(data_path)
    if not set(FEATURE_COLUMNS).issubset(data.columns):
        data = create_features(data)
    if target not in data.columns:
        raise ValueError(f"Target column {target!r} not found in {data_path}")
    X, y = data[FEATURE_COLUMNS], data[target]
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def tune_model(model_name: str, trials: int = 50, seed: int = 42, target: str = TARGET_COLUMN, data_path: Path = PROCESSED_DATA_PATH) -> dict:
    seed_everything(seed)
    X_train, X_test, y_train, y_test = load_training_data(seed, target, data_path)

    def objective(trial: optuna.Trial) -> float:
        model = suggest_model(trial, model_name, random_state=seed)
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, probability)
        if mlflow is not None:
            with mlflow.start_run(run_name=f"tune_{model_name}_{target}_trial_{trial.number}", nested=True):
                mlflow.log_param("seed", seed)
                mlflow.log_param("target", target)
                mlflow.log_params(trial.params)
                mlflow.log_metric("roc_auc", score)
        return score

    if mlflow is not None:
        mlflow.set_experiment("user-engagement-prediction")
    study = optuna.create_study(direction="maximize", study_name=f"{model_name}_{target}_roc_auc")
    with mlflow.start_run(run_name=f"tune_{model_name}_{target}") if mlflow is not None else nullcontext():
        study.optimize(objective, n_trials=trials)
        best_model = suggest_fixed_model(model_name, study.best_params, seed=seed)
        best_model.fit(X_train, y_train)
        probability = best_model.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test, probability)
        if mlflow is not None:
            mlflow.log_param("seed", seed)
            mlflow.log_param("target", target)
            mlflow.log_params(study.best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model, artifact_path=f"{model_name}_{target}_best_model")

    ensure_directories([EXPERIMENT_DIR, MODEL_DIR, REPORT_DIR])
    stem = artifact_stem(f"{model_name}_tuned", target)
    joblib.dump({"model": best_model, "features": FEATURE_COLUMNS, "target": target, "data_path": str(data_path)}, MODEL_DIR / f"{stem}.joblib")
    (REPORT_DIR / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"model": model_name, "target": target, "best_value": study.best_value, "best_params": study.best_params, "metrics": metrics}


def suggest_fixed_model(model_name: str, params: dict, seed: int = 42):
    class_weight = {"class_weight": "balanced"} if model_name == "random_forest" else {}
    if model_name == "random_forest":
        return RandomForestClassifier(**params, **class_weight, random_state=seed, n_jobs=-1)
    if model_name == "xgboost":
        return XGBClassifier(**params, eval_metric="logloss", random_state=seed)
    if model_name == "lightgbm":
        return LGBMClassifier(**params, random_state=seed, verbose=-1)
    raise ValueError(f"Unsupported model: {model_name}")


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def save_best_params(results: list[dict], seed: int = 42, target: str = TARGET_COLUMN) -> None:
    ensure_directories([EXPERIMENT_DIR])
    prefix = target_prefix(target)
    best_params_path = EXPERIMENT_DIR / f"{prefix}best_params.json"
    best_model_path = EXPERIMENT_DIR / f"{prefix}best_model.pkl"
    payload = {result["model"]: result for result in results}
    best_params_path.write_text(json.dumps(payload, indent=2))
    best_result = max(results, key=lambda result: result["best_value"])
    best_artifact = MODEL_DIR / f"{artifact_stem(best_result['model'] + '_tuned', target)}.joblib"
    joblib.dump(joblib.load(best_artifact), best_model_path)
    metadata_path = EXPERIMENT_DIR / f"{prefix}model_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update({"model_name": best_result["model"], "target": target, "model_version": f"seed-{seed}-{target}-{best_result['model']}", "seed": seed, "auc": round(float(best_result["best_value"]), 4)})
    metadata_path.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune tree-based engagement prediction models with Optuna.")
    parser.add_argument("--model", choices=TUNABLE_MODELS + ["all"], default="all")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--target", default=TARGET_COLUMN)
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_names = TUNABLE_MODELS if args.model == "all" else [args.model]
    results = [tune_model(model_name, trials=args.trials, seed=args.seed, target=args.target, data_path=args.data) for model_name in model_names]
    save_best_params(results, seed=args.seed, target=args.target)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
