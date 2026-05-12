# Experiment Log

## Baseline Plan

| Experiment | Model | Notes | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|---:|
| 001 | Logistic Regression | Balanced class weights | TBD | TBD | TBD | TBD |
| 002 | Random Forest | Non-linear baseline | TBD | TBD | TBD | TBD |
| 003 | XGBoost | Final candidate | TBD | TBD | TBD | TBD |
| 004 | LightGBM | Fast boosting alternative | TBD | TBD | TBD | TBD |

## Evaluation Notes

- Compare recall and precision jointly because retention campaigns need to catch at-risk users without over-notifying engaged users.
- Track feature importance and SHAP outputs for retention strategy recommendations.
- Run feature ablation after establishing a stable baseline.
