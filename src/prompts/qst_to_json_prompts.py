"""
This module contains all classes need to build prompts for the Topic Modeling task.
"""

class QstToJsonSystemPrompt:
    ROLE_DEFINITION = """You are a JSON converter."""

    INPUT_FORMAT_DEFINITION = """\nThe user will ask you to convert the questionnaire text into a JSON."""

    TASK_DEFINITION = """\nYou must reply with only the JSON which has the following structure: """
    OUTPUT_FORMAT_DEFINITION = """
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
        - Each answer object in the '_TF_ANSWERS' array has a single property 'ANSWER', which is a string representing the content of the answer."""
    
    QUESTION_TYPES_DESCRIPTION = """The admited question types are:
        - ID: 1 - Use this type of question to choose one answer from a list.
        - ID: 2 - Use this type of question to choose more answers from a list.
        - ID: 3 - Use this type of question to rate something.
        - ID: 4 - Use this type of question to aquire feedback from open questions."""

    FURTHER_SPECIFICATIONS = """
The enumeration of questions can be broken, so fix it while converting it.
The '[ ]' does not stand for a multi-choise question, so use the meaning of the question to decide its type.
The '______________________" stands for an open question. 
When the question is open, then '_TF_ANSWERS' is not needed."""


class QstToJsonUserPrompt:
    STANDARD_USER = """QUESTIONNAIRE TO CONVERT: \n%s"""

    SAMPLE_QUESTIONNAIRE_TXT = """
Access to Technology and Tools
1. What is your role in the company?
   - [ ] Executive/Senior Management
   - [ ] Manager
   - [ ] Staff/Employee
   - [ ] Intern
   - [ ] Other (please specify): ________________

2. How long have you been with the company?
   - [ ] Less than 1 year
   - [ ] 1-3 years
   - [ ] 3-5 years
   - [ ] 5-10 years
   - [ ] More than 10 years

3. On a scale of 1 to 5, how satisfied are you with the technology and tools provided by the company to perform your job? (1 being very dissatisfied, 5 being very satisfied)
   - [ ] 1
   - [ ] 2
   - [ ] 3
   - [ ] 4
   - [ ] 5

5. Do you have access to the necessary software and applications needed to perform your job effectively?
   - [ ] Yes
   - [ ] No
   - [ ] Partially

5. How would you rate the reliability and performance of the technology and tools provided by the company?
   - [ ] Excellent
   - [ ] Good
   - [ ] Average
   - [ ] Poor
   - [ ] Very poor

6. On a scale of 1 to 5, how easy is it for you to access technical support when encountering issues with technology or tools? (1 being very difficult, 5 being very easy)
   - [ ] 1
   - [ ] 2
   - [ ] 3
   - [ ] 4
   - [ ] 5

7. What improvements would you suggest to enhance access to technology and tools within the company?
    - __________________________________________________________________________

8. How can the company better support employees in utilizing technology and tools effectively?
    - __________________________________________________________________________

10. Any additional comments or feedback regarding access to technology and tools?
    - __________________________________________________________________________
"""


class QstToJsonAssistantPrompt:
    SAMPLE_CONVERTED_JSON = """
{
    "data": {
        "TF_QUESTIONNAIRES": [
            {
                "CODE": "ACCESS_TECHNOLOGY_TOOLS",
                "NAME": "Access to Technology and Tools",
                "TYPE_ID": 3,
                "_TF_QUESTIONS": [
                    {
                        "CODE": "Q1",
                        "NAME": "What is your role in the company?",
                        "TYPE_ID": 1,
                        "DISPLAY_ORDER": 1,
                        "_TF_ANSWERS": [
                            {"ANSWER": "Executive/Senior Management"},
                            {"ANSWER": "Manager"},
                            {"ANSWER": "Staff/Employee"},
                            {"ANSWER": "Intern"},
                            {"ANSWER": "Other"}
                        ]
                    },
                    {
                        "CODE": "Q2",
                        "NAME": "How long have you been with the company?",
                        "TYPE_ID": 1,
                        "DISPLAY_ORDER": 2,
                        "_TF_ANSWERS": [
                            {"ANSWER": "Less than 1 year"},
                            {"ANSWER": "1-3 years"},
                            {"ANSWER": "3-5 years"},
                            {"ANSWER": "5-10 years"},
                            {"ANSWER": "More than 10 years"}
                        ]
                    },
                    {
                        "CODE": "Q3",
                        "NAME": "On a scale of 1 to 5, how satisfied are you with the technology and tools provided by the company to perform your job? (1 being very dissatisfied, 5 being very satisfied)",
                        "TYPE_ID": 1,
                        "DISPLAY_ORDER": 3,
                        "_TF_ANSWERS": [
                            {"ANSWER": "1"},
                            {"ANSWER": "2"},
                            {"ANSWER": "3"},
                            {"ANSWER": "4"},
                            {"ANSWER": "5"}
                        ]
                    },
                    {
                        "CODE": "Q4",
                        "NAME": "Do you have access to the necessary software and applications needed to perform your job effectively?",
                        "TYPE_ID": 1,
                        "DISPLAY_ORDER": 4,
                        "_TF_ANSWERS": [
                            {"ANSWER": "Yes"},
                            {"ANSWER": "No"},
                            {"ANSWER": "Partially"}
                        ]
                    },
                    {
                        "CODE": "Q5",
                        "NAME": "How would you rate the reliability and performance of the technology and tools provided by the company?",
                        "TYPE_ID": 1,
                        "DISPLAY_ORDER": 5,
                        "_TF_ANSWERS": [
                            {"ANSWER": "Excellent"},
                            {"ANSWER": "Good"},
                            {"ANSWER": "Average"},
                            {"ANSWER": "Poor"},
                            {"ANSWER": "Very poor"}
                        ]
                    },
                    {
                        "CODE": "Q6",
                        "NAME": "On a scale of 1 to 5, how easy is it for you to access technical support when encountering issues with technology or tools? (1 being very difficult, 5 being very easy)",
                        "TYPE_ID": 1,
                        "DISPLAY_ORDER": 6,
                        "_TF_ANSWERS": [
                            {"ANSWER": "1"},
                            {"ANSWER": "2"},
                            {"ANSWER": "3"},
                            {"ANSWER": "4"},
                            {"ANSWER": "5"}
                        ]
                    },
                    {
                        "CODE": "Q7",
                        "NAME": "What improvements would you suggest to enhance access to technology and tools within the company?",
                        "TYPE_ID": 4,
                        "DISPLAY_ORDER": 7
                    },
                    {
                        "CODE": "Q8",
                        "NAME": "How can the company better support employees in utilizing technology and tools effectively?",
                        "TYPE_ID": 4,
                        "DISPLAY_ORDER": 8
                    },
                    {
                        "CODE": "Q9",
                        "NAME": "Any additional comments or feedback regarding access to technology and tools?",
                        "TYPE_ID": 4,
                        "DISPLAY_ORDER": 9
                    }
                ]
            }
        ]
    }
}"""
