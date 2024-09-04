Model Card
==============================

This work explores the capabilities of LLMs in the HR Questionnaire Generation task. Currently, we tested:
- `GPT-3.5-Turbo`
- `GPT-4-Turbo`

We employed them as is, thus no training was performed.

> [!IMPORTANT]
> In the adopted experiment naming convention, the models are identified by their deplument name on Azure.

### Task definition

This study explores the capabilities of GPT models in generating HR surveys, focusing on two task variants:

1. The user requests the model to generate a questionnaire by specifying the questionnaire topic and the number of questions.

2. The user requests the model to generate a questionnaire by specifying only the questionnaire topic.

These task definitions impose minimal constraints on the content to be generated, providing the model with a significant degree of freedom to demonstrate its creativity. However, the challenge lies in the limited information provided, which requires the model to rely heavily on its internal knowledge, increasing the risk of generating irrelevant or inaccurate content.

> [!IMPORTANT]
> In the adopted experiment naming convention, the task variant 1 is represented by the string `FULL`, while for the second one it is absent.


## Configuration

The experiments utilized Azure OpenAI APIs to deploy these models. The specific configurations for GPT-3.5-Turbo and GPT-4-Turbo on Azure are summarized below:
| Configuration                            | GPT-3.5-Turbo | GPT-4-Turbo  |
| :---                                     | :---:         |    :---:     |
| Version                                  | `0301`        |`1106-Preview`|
| Tokens per minute rate limit (thousands) | 120           | 30           |
| Rate limit (tokens per minute)           | 120,000       | 30,000       |
| Rate limit (requests per minute)         | 720           | 180          |

We employed both zero-shot and one-shot techniques to prompt the GPT models.

> [!IMPORTANT]
> In the adopted experiment naming convention, `Os` is used to denote experiments that used the zero-shot technique, while `1s` for those using the one-shot one.


The experimental setup involved testing various hyperparameter configurations for the GPT models:
- **Temperature**, whose tested values are {0, 0.25, 0.5}. 
- **Frequency penalty**, whose tested values are {0, 0.5, 1}.

The followin figure outlines the tested combinations (green entries) and the discarded ones (red entries):

<p align="center">
  <img src="/figures/params.png" alt="params">
</p>

Additionally, the following parameters were consistently configured across all experimental setups:
- **Max tokens**: for GPT-3.5-Turbo, the limit was set at 6,000 tokens, and for GPT-4-Turbo, it was set at 4,000 tokens.
- **Response format**, only available for GPT-4-Turbo, the response format was configured to output a valid JSON object.

> [!IMPORTANT]
> In the adopted experiment naming convention:
> - `9999MT` indicates the value of the max tokens (MT)
> - `9.99T` indicates the value of the temperature (T)
> - `9.99FP` indicates the value of the frequency penalty (FP)
> - `JSON` is appended for the experiments that specified the response format to the JSON type.

Further details about the adopted hyperparameter and their configuration in the API call can be found in the [documentation](https://platform.openai.com/docs/api-reference/chat/create) provided by OpenAI. 


## Evaluation

TODO

### Results

TODO
