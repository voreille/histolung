# HistoLung

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## TODO

- [ ] Use HistoQC instead of PyHIST for tiles selection
- [ ] Handle carefully magnifications fore each WSIsi
- [ ] Add more unit tests for module Y.
- [ ] Add attention channel for multilabel output?
- [ ] Implement multi-gpu support with torch.nn.DistributedDataParallel
- [ ] make usage of the word tile or patch consistent
- [ ] set the seeds!
- [ ] organize better the paths for multiple models and datasets
- [ ] Include MLFlow
- [ ] Implement a system to map embeddings with a set of parameters

Task: output csv file with tiles location
Goal: during tiling output a csv with the position of the tiles in the original WSI
Current Progress: 
- None

Task: check tissue folding tcga_lusc ID:TCGA-21-1080-01Z-00-DX1

Task: change histoqc config to make it work for ID (no mask): TCGA-44-7661-01Z-00-DX1

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         histolung and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── histolung   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes histolung a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## codebase organization
```
LungHistoMIL/
│
├── config/                         # Configuration files
│   ├── model_config.yaml
│
├── data/                           # Data processing and loading scripts
│   ├── dataset.py
│
├── models/                         # Models and wrappers
│   ├── __init__.py
│   ├── base_model.py               # Base wrapper class for models
│   ├── unified_mil_model.py        # Unified model with feature extraction and aggregation
│   ├── model_factory.py            # Factory for model creation
│
├── mil/                            # MIL training and evaluation
│   ├── mil_trainer.py              # MIL training and testing loop
│   ├── loss.py                     # MIL loss functions
│   ├── metrics.py                  # Custom metrics for MIL
│
├── explainability/                 # Explainable AI methods
│   ├── grad_cam.py                 # Grad-CAM implementation
│   ├── lime_explainer.py           # LIME implementation
│   ├── concept_explainer.py        # Concept-based methods (ACE, TCAV)
│   ├── explainer_factory.py        # Factory for creating explainer objects
│   ├── integrated_gradients.py     # Integrated Gradients implementation
│   ├── __init__.py
│
├── scripts/                        # Scripts to run experiments
│   ├── train.py                    # Entry point for training
│   ├── explain.py                  # Script to run explainability methods
│
├── utils/                          # Utility functions
│   ├── config_loader.py            # YAML configuration loader
│   ├── visualization.py            # Visualization tools for explainability
│
├── tests/                          # Unit and integration tests
│   ├── test_unified_mil_model.py
│   ├── test_mil_trainer.py
│   ├── test_explainability.py      # Tests for explainability methods
│
├── README.md                       # Project documentation
├── requirements.txt                # Dependencies
└── setup.py                        # Setup script
```
--------

