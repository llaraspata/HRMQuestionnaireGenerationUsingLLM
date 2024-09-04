HRM Questionnaire Generation by using LLMs
==============================
[![python](https://img.shields.io/badge/Python-3.11.5-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

This repository contains all the datasets, code, and supplementary materials to perform the Questionnaire Generation task in the Human Resource Management (HRM) domain by leveraging LLMs.
At the moment we focued on Surveys, that typically lack right/wrong or scored answers. Specifically, survey questionnaires are instrumental in gathering continuous feedback and opinions from employees, enabling organizations to monitor and enhance various aspects such as employee satisfaction and potential assessment.

Given the lack of adequate datasets, we built a new collection of HR Surveys. Details about the dataset can be found in the [Data Card](data/README.md).
We tested two GPT models (GPT-3.5-Turbo and GPT-4-Turbo) with different setting, in order to catch which are the factors that most contribute to an higher survey quality. Such details can be found in the [Model Card](models/README.md).
In our work, we designed a novel framework to automatically evaluate the generated content, due to the limitation of traditional metrics like raw ROUGE and BLEU. Thus, our metrics are able to estimate the quality of the surveys in terms of engagement, internal thematic variability, and flow. Further details are ported in the [Model Card](results/README.md).


[Notebooks](notebooks) show statistics on the new dataset, code sample usage, and the obtained results.



## Getting started

We recommend to use [Python 3.11.5](https://python.domainunion.de/downloads/release/python-3115/) to run our code, due to possible incompatibiities with newr versions.

### Installation
The installation process is described below:

1. Clone this repository:
   ```
   git clone https://github.com/llaraspata/HRMQuestionnaireGenerationUsingLLM.git
   ```
2. Create a virtual environment and activate it:
   ```
   python -m venv your_venv_name
   source <your_venv_name>/bin/activate  # On Windows, use: <your_venv_name>\Scripts\activate
   ```
3. Install all the project dependencies:
   ```
   pip install -r requirements.txt
   ```


### Experiments
The several experimental setting are configured in a [JSON file](src/models/experiment_config.json). To run all the configurations use the following command:
```
python -W ignore <path_to_repo_folder>/src/models/predict.py
```
Otherwise, to run a specific configuration use the following command:
```
python -W ignore <path_to_repo_folder>/src/models/predict.py --experiment-id "<experiment_id>" 
```

> [!NOTE]
> The option `-W ignore` allows to not display potential warnings during the script execution. To display them, just remove such an option.

### Evaluation
To run the the evaluation step execute the following command:
```
python -W ignore <path_to_repo_folder>/src/models/evaluate.py
```

### Utility
To perform the automatic conversion of the aumented data from unstructed text to JSON, run the following command:
```
python -W ignore <path_to_repo_folder>/src/data/convert_qst_to_json.py
```


## Citation

```bibtex
@misc{laraspata2024SurveyGeneration4HCM,
author = {Lucrezia Laraspata and Fabio Cardilli and Giovanna Castellano and Gennaro Vessio},
title = {Enhancing human capital management through GPT-driven questionnaire generation},
year = {2024},
url = {https://github.com/llaraspata/HRMQuestionnaireGenerationUsingLLM}
}
```




Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Raw questionnaires derived from the augmentation process.
    │   ├── interim        <- Intermediate augmented data that has been transformed to JSON.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The data used as starting point from this project
    │                         (taken from Talentia Software HCM).
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Predictions for each run experiments. For each of the a log and a picke file are saved.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── convert_qst_to_json.py
    │   │   └── TFQuestionnairesDataset.py
    │   │
    │   ├── prompts        <- Catalog of the employed prompts
    │   │   ├── qst_to_json_prompts.py
    │   │   ├── QstToJsonPromptGenerator.py
    │   │   ├── QstToJsonScenarioGenerator.py
    │   │   │
    │   │   ├── prediction_prompts.py
    │   │   ├── PredictionPromptGenerator.py
    │   │   ├── PredictionScenarioGenerator.py
    │   │   │
    │   │   ├── topic_modeling_prompts.py
    │   │   ├── TopicModelingPromptGenerator.py
    │   │   └── TopicModelingScenarioGenerator.py
    │   │
    │   ├── models         <- Scripts to run experiments and evaluations
    │   │   ├── experiment_config.json
    │   │   │
    │   │   ├── predict.py
    │   │   │
    │   │   ├── QuestionnairesEvaluator.py
    │   │   ├── ModelEvaluator.py
    │   │   └── evaluate.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       ├── experiment_pairs.json
    │       │
    │       ├── GlobalResultVisualizer.py
    │       ├── PairResultVisualizer.py
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
