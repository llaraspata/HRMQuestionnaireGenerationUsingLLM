HRM Questionnaire Generation by using LLMs
==============================
[![python](https://img.shields.io/badge/Python-3.11.5-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

This repository contains all the datasets, code, and supplementary materials to perform the Questionnaire Generation task in the Human Resource Management (HRM) domain by leveraging LLMs.
At the moment we focued on Surveys, that typically lack right/wrong or scored answers. Specifically, survey questionnaires are instrumental in gathering continuous feedback and opinions from employees, enabling organizations to monitor and enhance various aspects such as employee satisfaction and potential assessment.

Given the lack of adequate datasets, we built a new collection of HR Surveys. Details about the dataset can be found in the [Data Card](data/README.md).
We tested two GPT-3.5-Turbo, GPT-4-Turbo, LLaMa3, and Mixtral with different setting, in order to catch which are the factors that most contribute to an higher survey quality. Such details can be found in the [Model Card](models/README.md).
In our work, we designed a novel framework to automatically evaluate the generated content, due to the limitation of traditional metrics like raw ROUGE and BLEU. Thus, our metrics are able to estimate the quality of the surveys in terms of engagement, internal thematic variability, and flow. Further details are reported in the [Model Card](models/README.md).


[Notebooks](notebooks) show statistics on the new dataset, code sample usage, and the obtained results.



## 🚀 Getting started

We recommend to use [Python 3.11.5](https://python.domainunion.de/downloads/release/python-3115/) to run our code, due to possible incompatibiities with newr versions.

### 📥 Installation
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


### 🧪 Experiments
We tested models from GPT, LLaMa, and Mistral families. For each model family, we listed the experimental settings in the following JSON files: [GPT](src/models/GPT_experiment_config.json), [LLaMa](src/models/LLaMa_experiment_config.json), [Mistral](src/models/Mistral_experiment_config.json).


To run all the experiments for a model you can use the following command: 
```
python -W ignore <path_to_repo_folder>/src/models/predict.py --model "<model_name>"
```
Otherwise you can run single experiments using the command below:
```
python -W ignore <path_to_repo_folder>/src/models/predict.py --model "<model_name>" --experiment-id "<experiment_id>"
```
The `model` option is always mandatory and it can be equal to one of the following: `GPT` (or `gpt`), `LLaMa` (or `llama`), and `Mistral` (or `mistral`).

> [!CAUTION]
> **For GPT models**
>
> Make sure you have a valid (Azure) OpenAI access key, otherwise calling the OpenAI services will be forbidden. Then set it as an environment variable named `AZURE_OPENAI_KEY`.
>
> Moreover, note that we used a private deployment, so it cannot be accessed by users external to the Talentia HCM R&D Team. Thus, we recommed to substitue the `azure_endpoint` parameter value with a valid one in the API call.

> [!NOTE]
> The option `-W ignore` allows to not display potential warnings during the script execution. To display them, just remove such an option.

### 📊 Evaluation
To run the the evaluation step execute the following command:
```
python -W ignore <path_to_repo_folder>/src/models/evaluate.py
```
> [!NOTE]
> The command will launch the evaluation for all the experiments run for every model family.

### 🛠️ Utility
To perform the automatic conversion of the aumented data from unstructed text to JSON, run the following command:
```
python -W ignore <path_to_repo_folder>/src/data/convert_qst_to_json.py
```


## 🖋️ Citation

```bibtex
@inproceedings{laraspata2024qstgeneration,
   title={{Enhancing human capital management through GPT-driven questionnaire generation}},
   author={Laraspata, Lucrezia and Cardilli, Fabio and Castellano, Giovanna and Vessio, Gennaro},
   booktitle={Proceedings of the Eighth Workshop on Natural Language
    for Artificial Intelligence (NL4AI 2024) co-located with 23th
    International Conference of the Italian Association for Artificial
    Intelligence (AI*IA 2024)},
   year={2024},
   url = {https://uniroma2-my.sharepoint.com/:b:/g/personal/claudiudaniel_hromei_alumni_uniroma2_eu/EQn1_ibX2PRIiH9knF7fisUBYmTQpygq0MWaRKntADw6AA?e=Mjm5Ud}
}
```




📂 Project Organization
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
    ├── models             <- Predictions for each run experiments. For each of them a log and a picke file are saved.
    │
    ├── results            <- Evaluation results computed for each experiment.
    │
    ├── notebooks          <- Jupyter notebooks used to illustrate class usage, dataset insights, and experimental results.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
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
    │   ├── models         <- Scripts to run experiments
    │   │   ├── predict.py
    │   │   ├── utility.py
    │   │   │
    │   │   ├── GPT_experiment_config.json
    │   │   ├── gpt_predict.py
    │   │   │
    │   │   ├── LLaMa_experiment_config.json
    │   │   ├── llama_predict.py
    │   │   │
    │   │   ├── Mistral_experiment_config.json
    │   │   └── mistral_predict.py
    │   │
    │   ├── evaluation     <- Scripts to run evaluations
    │   │   ├── evaluate.py
    │   │   │
    │   │   ├── QuestionnairesEvaluator.py
    │   │   └── ModelEvaluator.py
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
