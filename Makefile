PYTHON ?= python3
TRIALS ?= 5
SEED ?= 42
DRY ?= 0
RUN = $(if $(filter 1,$(DRY)),@echo,@)

.PHONY: setup data train evaluate api docker test drift clean

setup:
	$(RUN) $(PYTHON) -m pip install -r requirements.txt

data:
	$(RUN) $(PYTHON) data/generate_synthetic_data.py
	$(RUN) $(PYTHON) src/data_connector.py --source csv --path data/raw/interaction_logs.csv

train:
	$(RUN) $(PYTHON) src/data_pipeline.py
	$(RUN) $(PYTHON) src/feature_engineering.py
	$(RUN) $(PYTHON) src/train.py --model random_forest --seed $(SEED)
	$(RUN) $(PYTHON) src/tune.py --model random_forest --trials $(TRIALS) --seed $(SEED)
	$(RUN) $(PYTHON) src/calibrate_threshold.py --metric f1

evaluate:
	$(RUN) $(PYTHON) src/compare_models.py
	$(RUN) $(PYTHON) src/feature_importance.py
	$(RUN) $(PYTHON) src/build_dashboard.py

api:
	$(RUN) uvicorn api.app:app --reload

docker:
	$(RUN) docker-compose up --build

test:
	$(RUN) pytest --cov=src --cov=api --cov-report=term-missing

drift:
	$(RUN) $(PYTHON) src/drift_monitor.py --new-data data/processed/engagement_features.csv
	$(RUN) $(PYTHON) src/retrain_trigger.py --threshold-auc 0.92 --dry-run

clean:
	$(RUN) find . -type d -name __pycache__ -prune -exec rm -rf {} +
	$(RUN) find . -type f -name '*.pyc' -delete
	$(RUN) rm -rf .pytest_cache .coverage htmlcov .pycache mlruns .cache
