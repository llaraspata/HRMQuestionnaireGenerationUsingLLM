import os
import pandas as pd
import time
import json
import os
from tqdm import tqdm
import argparse
import sys
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')
from src.data.TFQuestionnairesDataset import TFQuestionnairesDataset
from src.prompts.PredictionScenarioGenerator import PredictionScenarioGenerator
import src.models.utility as ut
import ollama


# -----------------
# Global variables
# -----------------
PROJECT_ROOT = os.getcwd()
VERSION_WITH_CONVERSATION = ["2.0", "2.1"]

# -----------------
# Main function
# -----------------
def main():
    conf = {
            "id": "0s_FULL_mistral-7x22B_4000MT_0T_0.1FP",
            "technique": "zero-shot",
            "k": 0,
            "has_full_params": True,
            "model": "mixtral:8x7b-instruct-v0.1-q2_K",#"llama3:8b-instruct-q4_0", #"mixtral:8x7b-instruct-v0.1-q2_K",
            "max_tokens": 4000,
            "temperature": 0,
            "frequency_penalty": 0.5,
            "prediction_path": "models"
        }
    
    dataset = TFQuestionnairesDataset()
    dataset.load_data(project_root=PROJECT_ROOT)
    print("- Dataset loaded!")
    
    system_prompt = """You are a Questionnaire Generator in the Human Resource Management field.
    The user will ask you to generate a questionnaire specifying the topic and the number of questions.
    If the user does not specify a valid topic, reply with "Sorry I cant help you".
    If the topic is valid, reply with only a JSON, which must respect the following format:
            - The root of the JSON is an object that contains a single property 'data'.
            - The 'data' property is an object that contains a single property 'TF_QUESTIONNAIRES'.
            - 'TF_QUESTIONNAIRES' is an array of only one element, which represents a questionnaire. It has the following properties:
                - 'CODE': (string) the questionnaire's code.
                - 'NAME': (string) the questionnaire's name.
                - 'TYPE_ID': (int) which is equal to 3.
                - '_TF_QUESTIONS': An array of objects, each representing a question.
            - Each question in the '_TF_QUESTIONS' array has the following properties:
                - 'CODE': (string) the question's unique code.
                - 'NAME': (string) the question's content.
                - 'TYPE_ID': (int) the question's type.
                - 'DISPLAY_ORDER': (int) the question's display order.
                - '_TF_ANSWERS': An array of objects, each representing a possible answer to the question.
            - Each answer object in the '_TF_ANSWERS' array has a single property 'ANSWER', which is a string representing the content of the answer.
    The admitted question's types are the following:
    - ID: 1, DESCRIPTION: Use this type of question to choose one answer from a list.
    - ID: 2, DESCRIPTION: Use this type of question to choose one or more answer from a list.
    - ID: 3, DESCRIPTION: Use this type of question to rate something.
    - ID: 4, DESCRIPTION: Use this type of question to aquire feedback.
    - ID: 5, DESCRIPTION: Use this type of question to reorder items.
    - ID: 6, DESCRIPTION: Use this type of question to disribute weights across several items/options.
    - ID: 7, DESCRIPTION: Use this type of question to clone questions from a template.
    - ID: 8, DESCRIPTION: Use this type of question when the answer to be given is a date or a date/time.
    Be creative and vary the syntax of your questions to enhance user engagement. Reply only with the JSON.
    """

    user_prompt = "Generate me a questionnaire on Stress at work with 7 questions"
    # Build messages and get LLM's response
    messages = ut.build_messages(conf["k"], system_prompt, [], [], user_prompt)

    # Record the start time
    start_time = time.time()                    

    if "response_format" in conf.keys():
        response = ollama.chat(
            model = conf["model"],
            messages=messages,
            options={
                "temperature": conf["temperature"],
                "repeat_penalty": conf["frequency_penalty"] + 1,
                "num_predict": conf["max_tokens"]
            },
            format=conf["response_format"],
            stream=False
        )
    else:
        response = ollama.chat(
            model = conf["model"],
            messages=messages,
            options={
                "temperature": conf["temperature"],
                "repeat_penalty": conf["frequency_penalty"] + 1,
                "num_predict": conf["max_tokens"]
            },
            stream=False
        )

    # Record the end time
    end_time = time.time()
    
    print("================================================")
    print(f"USR: {user_prompt}\n")
    
    prediction = response['message']['content']
    print(f"LLM: \n{prediction}")

    if response["done"]:
        prompt_tokens = response["prompt_eval_count"]
        completition_tokens = response["eval_count"]
        total_tokens = prompt_tokens + completition_tokens
    else:
        prompt_tokens = -1
        completition_tokens = -1
        total_tokens = -1
    time_spent = end_time - start_time

    print("-----------")
    print(f" [Spent time: {time_spent}]")
    print(f" [Prompt tokens: {prompt_tokens}]")
    print(f" [Completion tokens: {completition_tokens}]")
    print(f" [Total tokens: {total_tokens}]")
    
        
    
        
    print("================================================")



if __name__ == '__main__':
    main()