import os
import pandas as pd
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# -----------------
# Global variables
# -----------------
PREDICTION_COLUMNS = ["QUESTIONNAIRE_ID", "SAMPLE_QUESTIONNAIRES_IDS", "GROUND_TRUTH_JSON", "PREDICTED_JSON", "REPORTED_EXCEPTION", 
                      "RESPONSE_TIME", "PROMPT_TOKENS", "COMPLETITION_TOKENS", "TOTAL_TOKENS"]


# -----------------
# Methods
# -----------------
def save_df_to_folder(df, folder_path, file_name):
    """
        Saves a DataFrame to a folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, file_name)

    df.to_pickle(file_path)



def build_messages(k, system_prompt, sample_user_prompts, assistant_prompts, user_prompt):
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



def add_prediction(df, questionnaire_id, sample_questionnaire_ids=[], ground_truth="", prediction="", spent_time=0, 
                    prompt_tokens=0, completition_tokens=0, total_tokens=0, reported_exception=""):
    """
        Adds a prediction to the DataFrame.
    """
    new_row = pd.DataFrame({
        "QUESTIONNAIRE_ID": [questionnaire_id],
        "SAMPLE_QUESTIONNAIRES_IDS": [sample_questionnaire_ids], 
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


def load_mistral_llm(model_name):
    hf_key = os.getenv("HF_MISTRAL_MODELS")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=hf_key)

    config.max_position_embeddings = 8096

    quantization_config = BitsAndBytesConfig(
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        load_in_4bit=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        config=config,
        trust_remote_code=True,
        quantization_config=quantization_config,
        token=hf_key
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    # OSError: Can't load tokenizer for 'mistralai/Mistral-7B-Instruct-v0.2'. 
    # If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. 
    # Otherwise, make sure 'mistralai/Mistral-7B-Instruct-v0.2' is the correct path to a directory containing all relevant files for a LlamaTokenizerFast tokenizer.
    # (.env) llaraspata3@gandalf:~/HRMQuestionnaireGenerationUsingLLM$ 

    print(f"Loaded -> {model_name}!")

    return model, tokenizer