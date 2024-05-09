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
from src.prompts.ScenarioGenerator import ScenarioGenerator


# -----------------
# Global variables
# -----------------
PROJECT_ROOT = os.getcwd()
CONFIG_FILENAME = "experiment_config.json"
PREDICTION_COLUMNS = ["QUESTIONNAIRE_ID", "GROUND_TRUTH_JSON", "PREDICTED_JSON", "REPORTED_EXCEPTION", 
                      "RESPONSE_TIME", "PROMPT_TOKENS", "COMPLETITION_TOKENS", "TOTAL_TOKENS"]


# -----------------
# Main function
# -----------------
def main(args):

    experiment_id = args.experiment_id

    client = AzureOpenAI(
        azure_endpoint = "https://openai-hcm-dev-d06.openai.azure.com/", 
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version="2024-02-15-preview"
    )

    # Load data
    dataset = TFQuestionnairesDataset()
    dataset.load_data(project_root=PROJECT_ROOT)

    # Read the experiment configuration to be tested
    config_path = os.path.join(PROJECT_ROOT, "src", "models", CONFIG_FILENAME)
    with open(config_path, "r") as f:
        experiment_confs = json.load(f)
    f.close()

    for conf in experiment_confs["configs"]:
        if experiment_id is not None and conf["id"] != experiment_id:
            continue

        # Create the experiment run directory
        run_dir = os.path.join(PROJECT_ROOT, "models", conf["id"])
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        
        # Set output file names
        predictions_filename = "predictions.pkl"
        log_filename = "log.txt"
    
        # Run the experiment        
        result_df = _run_experiment(client=client, dataset=dataset, conf=conf, run_dir=run_dir, log_filename=log_filename)
        
        # Save the results
        _save_df_to_folder(result_df, run_dir, predictions_filename)



# -----------------
# Helper functions
# -----------------
def _run_experiment(client, dataset, conf, run_dir, log_filename):

    print("================================================")
    print(f"Running experiment: {conf['id']}")

    # Set the scenario for the current experiment
    scenario = ScenarioGenerator(experiment_config=conf, dataset=dataset)

    # Create the DataFrame for the predictions and other statistics
    predictions_df = pd.DataFrame(columns=PREDICTION_COLUMNS)

    spent_secs_per_request = []
    count_reported_exceptions = 0

    n_questionnaires = len(dataset.questionnaires["ID"])

    log_path = os.path.join(run_dir, log_filename)
    with open(log_path, "a") as log_file: 
        
        for i in tqdm(range(n_questionnaires)):
            try:
                # Generate prompts
                questionnaire_id = dataset.questionnaires["ID"][i]
                system_prompt, sample_user_prompts, assistant_prompts, user_prompt = scenario.generate_scenario(log_file=log_file,
                                                                                                                        current_questionnaire_id=questionnaire_id, dataset=dataset)
            
                log_file.write("\n-------------------")
                log_file.write("\n[PROMPTS]")
                log_file.write(f"\n     - System: \n{system_prompt}")
                log_file.write(f"\n     - User: \n{user_prompt}")
                log_file.write("\n-------------------")

                # Build messages and get LLM's response
                messages = _build_messages(conf["k"], system_prompt, sample_user_prompts, assistant_prompts, user_prompt)

                # Record the start time
                start_time = time.time()

                if "response_format" in conf.keys():
                    response = client.chat.completions.create(
                        model = conf["model"],
                        messages=messages,
                        temperature=conf["temperature"],
                        max_tokens=conf["max_tokens"],
                        frequency_penalty=conf["frequency_penalty"],
                        response_format=conf["response_format"]
                    )
                else:
                    response = client.chat.completions.create(
                        model = conf["model"],
                        messages=messages,
                        temperature=conf["temperature"],
                        max_tokens=conf["max_tokens"],
                        frequency_penalty=conf["frequency_penalty"]
                    )

                # Record the end time
                end_time = time.time()

                ground_truth = dataset.to_json(questionnaire_id)
                prediction = response.choices[0].message.content

                prompt_tokens = response.usage.prompt_tokens
                completition_tokens = response.usage.completion_tokens
                total_tokens = prompt_tokens + completition_tokens

                log_file.write("\n-------------------")
                log_file.write("\n[LLM ANSWER]\n")
                log_file.write(prediction)
                log_file.write("\n-------------------")
    
                # Compute the spent time
                time_spent = end_time - start_time
                spent_secs_per_request.append(time_spent)

                predictions_df = _add_prediction(df=predictions_df, questionnaire_id=questionnaire_id, ground_truth=ground_truth, prediction=prediction, spent_time=time_spent, 
                                                 prompt_tokens=prompt_tokens, completition_tokens=completition_tokens, total_tokens=total_tokens)
    
                time.sleep(2)  # sleep for 2 seconds to avoid exceeding the OpenAI API rate limit or other kind of errors
                    
            except Exception as e:
                end_time = time.time()
                time_spent = end_time - start_time
                spent_secs_per_request.append(time_spent)

                count_reported_exceptions += 1
                predictions_df = _add_prediction(df=predictions_df, questionnaire_id=questionnaire_id, spent_time=time_spent, reported_exception=e)
                
    print("-------------------")
    print("[TIME]")
    if len(spent_secs_per_request) > 0:
        total_time = sum(spent_secs_per_request)
        print(f"     - Avg per request: {total_time / len(spent_secs_per_request)}")
        print(f"     - Total: {total_time}")
        
    else:
        print("No available data about the time spent per request.")
    print("-------------------")

    print("-------------------")
    print("[TEST RESULTS]")
    print(f"     - Completed successfully: {n_questionnaires - count_reported_exceptions}")
    print(f"     - Reported exception: {count_reported_exceptions}")
    print("-------------------")

    print("================================================")


    return predictions_df


def _build_messages(k, system_prompt, sample_user_prompts, assistant_prompts, user_prompt):
    """
        Builds the messages to be sent to the LLM.
    """
    messages = []

    messages.append({"role": "system", "content": system_prompt})

    for i in range(k):
        messages.append({"role": "user", "content": sample_user_prompts[i][0]})
        messages.append({"role": "assistant", "content": assistant_prompts[i][0]})

    messages.append({"role": "user", "content": user_prompt})

    return messages


def _add_prediction(df, questionnaire_id, ground_truth="", prediction="", spent_time=0, 
                    prompt_tokens=0, completition_tokens=0, total_tokens=0, reported_exception=""):
    """
        Adds a prediction to the DataFrame.
    """
    new_row = pd.DataFrame({
        "QUESTIONNAIRE_ID": [questionnaire_id],
        "GROUND_TRUTH_JSON": [ground_truth],
        "PREDICTED_JSON": [prediction],
        "REPORTED_EXCEPTION": [reported_exception],
        "RESPONSE_TIME": [spent_time],
        "PROMPT_TOKENS": [prompt_tokens],
        "COMPLETITION_TOKENS": [completition_tokens],
        "TOTAL_TOKENS": [total_tokens],
    })

    df = pd.concat([df, new_row], ignore_index=True)

    return df


def _save_df_to_folder(df, folder_path, file_name):
    """
        Saves a DataFrame to a folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, file_name)

    df.to_pickle(file_path)



if __name__ == '__main__':
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--experiment-id', type=str, help='Name of the experiment to run')
    
    main(aparser.parse_args())