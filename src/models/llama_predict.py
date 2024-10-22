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
CONFIG_FILENAME = "LLaMa_experiment_config.json"
VERSION_WITH_CONVERSATION = ["2.0", "2.1"]

# -----------------
# Main function
# -----------------
def main(args):

    if args is not None and args.experiment_id is not None:
        experiment_id = args.experiment_id
    else:
        experiment_id = None

    model = "LLaMa"
    task = "Survey"

    if args is not None and args.prompt_version is not None:
        prompt_version = args.prompt_version
    else:
        prompt_version = "1.0"

    # Create the prompt version and model directory
    setting_dir = os.path.join(PROJECT_ROOT, "models", task, prompt_version, model)
    if not os.path.exists(setting_dir):
        os.makedirs(setting_dir)

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
        run_dir = os.path.join(setting_dir, conf["id"])
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        
        # Set output file names
        predictions_filename = "predictions.pkl"
        log_filename = "log.txt"

        # Run the experiment        
        result_df = _run_experiment(dataset=dataset, conf=conf, prompt_version=prompt_version, run_dir=run_dir, log_filename=log_filename)
        
        # Save the results
        ut.save_df_to_folder(result_df, run_dir, predictions_filename)


# -----------------
# Helper functions
# -----------------
def _run_experiment(dataset, conf, prompt_version, run_dir, log_filename):

    print("================================================")
    print(f"Running experiment: {conf['id']}")

    # Set the scenario for the current experiment
    scenario = PredictionScenarioGenerator(project_root=PROJECT_ROOT, prompt_version=prompt_version, experiment_config=conf, dataset=dataset)

    # Create the DataFrame for the predictions and other statistics
    predictions_df = pd.DataFrame(columns=ut.PREDICTION_COLUMNS)

    spent_secs_per_request = []
    count_reported_exceptions = 0

    n_questionnaires = len(dataset.questionnaires["ID"])

    log_path = os.path.join(run_dir, log_filename)
    with open(log_path, "w") as log_file: 
        
        for i in tqdm(range(n_questionnaires)):
            try:
                # Generate prompts
                questionnaire_id = dataset.questionnaires["ID"][i]
                system_prompt, sample_user_prompts, assistant_prompts, user_prompt, sample_questionnaires_ids = scenario.generate_scenario(log_file=log_file,
                                                                                                                current_questionnaire_id=questionnaire_id, dataset=dataset)
            
                log_file.write("\n-------------------")
                log_file.write("\n[PROMPTS]")
                log_file.write(f"\n     - System: \n{system_prompt}")
                log_file.write(f"\n     - User: \n{user_prompt}")
                log_file.write("\n-------------------")

                # Build messages and get LLM's response
                messages = ut.build_messages(conf["k"], system_prompt, sample_user_prompts, assistant_prompts, user_prompt)

                # Record the start time
                start_time = time.time()                    

                if "response_format" in conf.keys():
                    response = ollama.chat(
                        model = conf["model"],
                        messages=messages,
                        options={
                            "temperature": conf["temperature"]/2,
                            "repeat_penalty": conf["frequency_penalty"],
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
                            "temperature": conf["temperature"]/2,
                            "repeat_penalty": conf["frequency_penalty"],
                            "num_predict": conf["max_tokens"]
                        },
                        stream=False
                    )

                # Record the end time
                end_time = time.time()

                ground_truth = dataset.to_json(questionnaire_id)
                prediction = response['message']['content']
                time_spent = end_time - start_time

                ground_truth_content = ""
                prediction_json = ""

                if VERSION_WITH_CONVERSATION.__contains__(prompt_version):
                    convert_user_prompt = scenario.generate_last_user()
                    messages = ut.append_messages(messages=messages, assistant_reply=prediction, user_prompt=convert_user_prompt)

                    start_time = time.time()
                    
                    if "response_format" in conf.keys():
                        response = ollama.chat(
                            model = conf["model"],
                            messages=messages,
                            options={
                                "temperature": conf["temperature"]/2,
                                "repeat_penalty": conf["frequency_penalty"],
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
                                "temperature": conf["temperature"]/2,
                                "repeat_penalty": conf["frequency_penalty"],
                                "num_predict": conf["max_tokens"]
                            },
                            stream=False
                        )

                    end_time = time.time()

                    time_spent = time_spent + (end_time - start_time)
                    ground_truth_content = dataset.to_text(questionnaire_id)
                    prediction_json = response['message']['content']

                if response["done"]:
                    prompt_tokens = response["prompt_eval_count"]
                    completition_tokens = response["eval_count"]
                    total_tokens = prompt_tokens + completition_tokens
                else:
                    prompt_tokens = -1
                    completition_tokens = -1
                    total_tokens = -1
                    
                log_file.write("\n-------------------")
                log_file.write("\n[LLM ANSWER]\n")
                log_file.write(prediction)

                if VERSION_WITH_CONVERSATION.__contains__(prompt_version):
                    log_file.write(f"\n     - User: \n{user_prompt}")
                    log_file.write(f"\n[LLM ANSWER] -> JSON \n{prediction}")
                
                log_file.write("\n-------------------")
    
                # Compute the spent time
                spent_secs_per_request.append(time_spent)

                if not VERSION_WITH_CONVERSATION.__contains__(prompt_version):
                    predictions_df = ut.add_prediction(df=predictions_df, questionnaire_id=questionnaire_id, sample_questionnaire_ids=sample_questionnaires_ids,
                                                       ground_truth_json=ground_truth, prediction_json=prediction, spent_time=time_spent,
                                                       prompt_tokens=prompt_tokens, completition_tokens=completition_tokens, total_tokens=total_tokens)
                elif prompt_version == "1.1":
                    explanation, prediction_json = ut.extract_explanation_and_json(response_text=prediction)
                    predictions_df = ut.add_prediction(df=predictions_df, questionnaire_id=questionnaire_id, sample_questionnaire_ids=sample_questionnaires_ids,
                                                       ground_truth_content=ground_truth_content, prediction_content=explanation,
                                                       ground_truth_json=ground_truth, prediction_json=prediction_json,
                                                       spent_time=time_spent, prompt_tokens=prompt_tokens, completition_tokens=completition_tokens, total_tokens=total_tokens)
                else:
                    predictions_df = ut.add_prediction(df=predictions_df, questionnaire_id=questionnaire_id, sample_questionnaire_ids=sample_questionnaires_ids,
                                                       ground_truth_content=ground_truth_content, prediction_content=prediction,
                                                       ground_truth_json=ground_truth, prediction_json=prediction_json,
                                                       spent_time=time_spent, prompt_tokens=prompt_tokens, completition_tokens=completition_tokens, total_tokens=total_tokens)
                
                time.sleep(2)
                
            except Exception as e:
                end_time = time.time()
                time_spent = end_time - start_time
                spent_secs_per_request.append(time_spent)

                count_reported_exceptions += 1
                predictions_df = ut.add_prediction(df=predictions_df, questionnaire_id=questionnaire_id, spent_time=time_spent, reported_exception=e)
                
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


if __name__ == '__main__':
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--prompt-version', type=str, help='The version of the prompt to use')
    aparser.add_argument('--experiment-id', type=str, help='Name of the experiment to run')
    aparser.add_argument('--prompt-version', type=str, help='The version of the prompt to use')
    
    main(aparser.parse_args())
