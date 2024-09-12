import os
import pandas as pd
import time
import json
import os
from tqdm import tqdm
import argparse
from openai import AzureOpenAI
import sys
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')
from src.data.TFQuestionnairesDataset import TFQuestionnairesDataset
from src.prompts.PredictionScenarioGenerator import PredictionScenarioGenerator
from src.models.gpt_predict import main as gpt
from src.models.mistral_predict import main as mistral



PROJECT_ROOT = os.getcwd()


def main(args):

    model = args.model
    experiment_id = args.experiment_id

    if model == "gpt" or model == "GPT":
        gpt(experiment_id)
    elif model == "mistral" or model == "Mistral":
        mistral(experiment_id)
    else:
        print(f"Specify a valid model name. '{model}' is not a supported.")


if __name__ == '__main__':
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--prompt-version', type=str, help='The version of the prompt to use')
    aparser.add_argument('--model', type=str, help='Name of the model family to use')
    aparser.add_argument('--experiment-id', type=str, help='Name of the experiment to run')
    
    main(aparser.parse_args())