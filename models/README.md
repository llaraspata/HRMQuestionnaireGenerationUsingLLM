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

We propose a new evaluation framework that could automatically estimate the quality of the generated surveys. The main metrics are described below. 

> [!NOTE]
> For completeness, in the repository we kept all the computed metrics during our experiments, although they might be poorly informative.

### Intra-questionnaire similarity
An engaging questionnaire typically features high lexical variability, which prevents it from becoming monotonous or tedious.

**ROUGE-L** was calculated for all pairs of generated questions and then averaged. Higher scores indicate that the questions share nearly identical syntactic and lexical structures, ultimately leading to a lower overall questionnaire quality.


### Semantic Similarity
we specifically designed a score that evaluates the similarity between generated questions, ground-truth questions, and the overall questionnaire topic while penalizing deviations from the ideal question order. The defined score, SemSim, is formalized as follows:
```math
    \text{SemSim} = \frac{\alpha \cdot sim(G, H) + \beta \cdot sim(G, T)}{(\alpha + \beta) - dev( pos(G), pos(H))},
```
where $G$ indicates the generated question, $H$ the human-written question, $T$ the questionnaire topic, $sim( X, Y)$ indicates the semantic similarity between elements $X$ and $Y$, calculated using cosine similarity on their embeddings, $\alpha$ is the weight assigned to the similarity between the generated question $G$ and the human-written question $H$, $\beta$ is the weight assigned to the similarity between the generated question $G$ and the questionnaire topic $T$, $pos(X)$ indicates the position of question $X$ in its respective questionnaire, and $dev( pos(G), pos(H))$ represents the normalized position deviation of the generated question $G$ from the ideal position, given by the human-written question $H$. This deviation is computed as follows:
```math
        \frac{|pos(G) - pos(H)|}{\max(N, M)},
```
where $N$ is the number of questions in the generated questionnaire, and $M$ is the number of questions in the ground-truth questionnaire. This deviation ranges from 0 to 1, with scores closer to 0 indicating that the model generated the question in the correct position and scores closer to 1 indicating significant deviation. SemSim ranges between 0 and 1. Lower scores suggest low weighted cosine similarity or high position deviation, while higher scores indicate substantial similarity and minimal deviation.


### Serendipity

Serendipity can be interpreted as the thematic variability within a single questionnaire in the context of questionnaire generation. This variability enriches the content and increases engagement by avoiding repetitive or overly focused questions. Inspired by Boldi et al.'s definition, we adapted the concept of serendipity for our study as follows:
```math
    \text{Serendipity} = \frac{n}{\min(C, R)}
```
where $n$ represents the number of generated questions relevant to the questionnaire topic, $C$ is the number of possible subtopics generally relevant to the main topic, and $R$ is the total number of generated questions. The serendipity score ranges from 0 to 1. A score closer to 1 indicates that almost every question addresses a different subtopic, contributing to high thematic variability. Conversely, a score closer to 0 suggests lower variability, increasing the risk of duplicate or redundant questions.


### Instruction alignment

The temperature and frequency penalty values variation influences the tokens sampled during the generation process. Increasing these values to encourage the model to be more variable and creative can degrade the quality of the generated JSON output. This degradation manifests in the model potentially omitting specified properties or generating text that does not adhere to JSON standards.

### Turing test

We selected 3 pairs of questionnaires to be submitted to HR professionals according to the computed metrics. The employed materials is contained in [this folder](/results/turing_test/submitted).

For each pair, participants were given 60 seconds to review the questionnaires and then asked to respond to the following questions:

- `<Topic>` - Which questionnaire is AI Generated? 
    - Questionnaire A;
    - Questionnaire B.
-  Why do you believe the questionnaire you chose was AI-generated? 
    - Variability of questions;
    - Variability of answers;
    - Variability of response types;
    - Language style;
    - Questions sequence/order;
    - Consistency between questions and related answers;
    - Relevance to topic.

The first question was single-choice, while the second was multi-choice. Although an open-ended question would have been preferable for deeper insights, the main goal was to maintain participants' interest and involvement without overwhelming them.