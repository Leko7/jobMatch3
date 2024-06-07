"""
This script does the following :

- Use GPT4 to identify the wanted soft-skills from a job description (job_description.txt), and store them into job_description_specific_traits.json.

- Translate this soft-skills into psychological traits, using a dictionary (skill_to_traits.json) built according to psychometrics research.

- Retrieve the psychological traits associated to the job name (non redundent with the traits detected in job_description_specific_traits.json),
using a dictionary (job_to_traits.json) built according to psychometrics research, and store the traits in missing_traits.json.

- Generate a html report which presents the detected traits from the job description (job_description_specific_traits.json),
and asks the user if he wants to add the other traits, related to the job name (missing_traits.json).

"""
# Load the .env file at the project root into environment variables
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()

from utils import select_traits_and_create_report

if __name__ == "__main__" :

    # Arguments to specify for each project
    job_name = 'Customer Success Manager'
    with_job_description = True
    manager_job = False

    select_traits_and_create_report(job_name, with_job_description, manager_job, client)