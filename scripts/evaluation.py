import json
import pandas as pd
# Load the .env file at the project root into environment variables
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()

from utils import select_traits_and_create_report

if __name__ == "__main__" :

    i = 2

    # Arguments to specify for each project
    job_name = 'commercial'
    with_job_description = True
    manager_job = False

    jobs = ["chefProjet",
            "commercial",
            "csm",
            "cto",
            "headMarketing",
            "ingénieur",
            "marketing",
            "rh",
            "technicoCommercial"
    ]

    for job in jobs:

        with open(f"evaluation/job_descriptions/{job}_{i}.txt", 'r') as file:
            job_description = file.read()

        with open("input_data/job_description.txt", "w") as file:
            file.write(job_description)

        select_traits_and_create_report(job_name, with_job_description, manager_job, client)

        # with open("output_data/job_soft_skills.json", "r") as file:
        #     job_soft_skills = json.load(file)

        # with open(f"evaluation/detected_skills/{job}_{i}.json",'w') as file:
        #     json.dump(job_soft_skills, file)

        with open(f"output_data/llm_step_1_output.json", "r") as file:
            detected_skills = json.load(file)

        with open(f"evaluation/detected_skills/{job}_{i}.json", "w") as file:
            json.dump(detected_skills, file)

        with open(f"output_data/job_soft_skills.json", "r") as file:
            recognized_skills = json.load(file)

        with open(f"evaluation/recognized_skills/{job}_{i}.json", "w") as file:
            json.dump(recognized_skills, file)


    with open('input_data/skill_to_traits.json', 'r') as file:
        skill_to_traits = json.load(file)

    df = pd.DataFrame()

    for job in jobs:

        with open(f"evaluation/detected_skills/{job}_{i}.json", "r") as file:
            detected_skills = json.load(file)

        with open(f"evaluation/job_descriptions/{job}_{i}.txt", "r") as file:
            job_description = file.read()

        with open(f"evaluation/recognized_skills/{job}_{i}.json", "r") as file:
            recognized_skills = json.load(file)

        job_row = {}

        job_row['job'] = job
        job_row['job_description'] = job_description
        job_row['skills_detected'] = detected_skills['soft_skills']
        job_row['skills_recognized'] = recognized_skills
        job_row['n_detected'] = len(job_row['skills_detected'])
        job_row['n_recognized'] = len(job_row['skills_recognized'])
        job_row['associated_traits'] = list(set([trait for skill in recognized_skills if skill in skill_to_traits for trait in skill_to_traits[skill] ]))
        job_row['n_traits'] = len(job_row['associated_traits'])
        
        df = pd.concat([df, pd.DataFrame([job_row])], ignore_index = True)
        
    df.to_csv(f"evaluation/evaluation_{i}.csv")
    print(f"Evaluation table saved to evaluation/evaluation_{i}.csv")