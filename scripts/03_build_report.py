"""
This script does the following :

- Uses results.csv (file that contains the results of candidates to the Big Five Personality Test) and selected_traits.json (wanted traits for the job,
obtained from the 01 and 02 scripts execution) to generate a report.

The report's purpose is to :

- assess if the personality of the candidate fits or misfits what is expected for the job
- suggest questions (located in trait_info.csv) depending on different cases (misfit and low level of trait, fit and low, misfit and high, fit and high).
- present the most marked traits of the candidate (highest and lowest scores), and suggest questions depending on cases (low level or high level). This part is unrelated to the job.

"""
from utils import get_report_data, generate_file

if __name__ == "__main__" :

    job_name = "commercial"
    candidate_mail = "laure.bourdon@haffner-energy.com"

    get_report_data(job_name, candidate_mail)
    generate_file()