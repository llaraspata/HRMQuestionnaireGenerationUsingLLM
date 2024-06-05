import os
import pandas as pd
import time
import os
from tqdm import tqdm
from openai import AzureOpenAI
import sys
from docx import Document
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')
from src.prompts.QstToJsonScenarioGenerator import QstToJsonScenarioGenerator


# -----------------
# Global variables
# -----------------
PROJECT_ROOT = os.getcwd()
MODEL = "gpt-35-turbo-dev"
CONVERSION_COLUMNS = ["QUESTIONNAIRE_ID", "JSON", "REPORTED_EXCEPTION", 
                      "RESPONSE_TIME", "PROMPT_TOKENS", "COMPLETITION_TOKENS", "TOTAL_TOKENS"]


# -----------------
# Main function
# -----------------
def main():
    conversions_df = pd.DataFrame(columns=CONVERSION_COLUMNS)
    client = AzureOpenAI(
        azure_endpoint = "https://openai-hcm-dev-d06.openai.azure.com/", 
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version="2024-02-15-preview"
    )

    external_data_path = os.path.join(PROJECT_ROOT, "data", "external")
    converted_dir = os.path.join(PROJECT_ROOT, "data", "interim")

    scenario_generator = QstToJsonScenarioGenerator()

    print("Converting questionnaires to JSON format...")
    id = 1
    for qst_docx in tqdm(os.listdir(external_data_path)):
        if not qst_docx.endswith(".docx"):
            continue
        
        qst_txt = _read_word_file(os.path.join(external_data_path, qst_docx))
        
        # Set output file names
        conversions_filename = "qst_to_json_result.pkl"
        log_filename = "log.txt"

        # Run the conversion        
        result_df = _run_experiment(questionnaire_id=id, scenario=scenario_generator, client=client, qst_text=qst_txt, run_dir=converted_dir, log_filename=log_filename)
        conversions_df = pd.concat([conversions_df, result_df], ignore_index=True)
        id += 1
    
    _save_df_to_folder(conversions_df, converted_dir, conversions_filename)
    print("Conversion completed.")



# -----------------
# Helper functions
# -----------------
def _read_word_file(file_path):
    doc = Document(file_path)
    content = []
    for para in doc.paragraphs:
        content.append(para.text)
    return "\n".join(content)


def _run_experiment(questionnaire_id, scenario, client, qst_text, run_dir, log_filename):
    # Create the DataFrame for the conversions and other statistics
    conversions_df = pd.DataFrame(columns=CONVERSION_COLUMNS)

    spent_secs_per_request = []
    count_reported_exceptions = 0

    log_path = os.path.join(run_dir, log_filename)
    with open(log_path, "a") as log_file: 
        try:
            # Generate prompts
            system_prompt, user_prompt = scenario.generate_scenario(questionnaire=qst_text)
        
            log_file.write("\n-------------------")
            log_file.write("\n[PROMPTS]")
            log_file.write(f"\n     - System: \n{system_prompt}")
            log_file.write(f"\n     - User: \n{user_prompt}")
            log_file.write("\n-------------------")

            # Build messages and get LLM's response
            messages = _build_messages(system_prompt, user_prompt)

            # Record the start time
            start_time = time.time()
        
            response = client.chat.completions.create(
                model = MODEL,
                messages=messages,
                temperature=0,
                max_tokens=6000,
                frequency_penalty=0
            )

            # Record the end time
            end_time = time.time()

            converted_qst = response.choices[0].message.content

            prompt_tokens = response.usage.prompt_tokens
            completition_tokens = response.usage.completion_tokens
            total_tokens = prompt_tokens + completition_tokens

            log_file.write("\n-------------------")
            log_file.write("\n[LLM ANSWER]\n")
            log_file.write(converted_qst)
            log_file.write("\n-------------------")

            # Compute the spent time
            time_spent = end_time - start_time
            spent_secs_per_request.append(time_spent)

            conversions_df = _add_converted_qst(df=conversions_df, questionnaire_id=questionnaire_id, converted_qst=converted_qst, spent_time=time_spent, 
                                                prompt_tokens=prompt_tokens, completition_tokens=completition_tokens, total_tokens=total_tokens)

            time.sleep(2)  # sleep for 2 seconds to avoid exceeding the OpenAI API rate limit or other kind of errors
                
        except Exception as e:
            end_time = time.time()
            time_spent = end_time - start_time
            spent_secs_per_request.append(time_spent)

            count_reported_exceptions += 1
            conversions_df = _add_converted_qst(df=conversions_df, questionnaire_id=questionnaire_id, spent_time=time_spent, reported_exception=e)
                
    return conversions_df


def _build_messages(system_prompt, user_prompt):
    """
        Builds the messages to be sent to the LLM.
    """
    messages = []

    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    return messages


def _add_converted_qst(df, questionnaire_id, converted_qst="", spent_time=0, 
                    prompt_tokens=0, completition_tokens=0, total_tokens=0, reported_exception=""):
    """
        Adds a prediction to the DataFrame.
    """
    new_row = pd.DataFrame({
        "QUESTIONNAIRE_ID": [questionnaire_id],
        "JSON": [converted_qst],
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
    main()