# JobMatch

Generate a personalized psychometric report, based on a job description (job_description.txt) and the psychometric results of a candidate to the Big Five personality test, stored in the results table (results.csv).

The first scripts uses GPT4 API calls to analyse the job description and determine the most relevant psychological traits.
It generates a report about the trait recognition (traits_recognition_report.html).

The second scripts adds relevant traits, based on the name of the job and the position type (ex: manager).

The third script creates the personalized report for a specific candidate, in pdf format (output_data/candidates_reports/candidate_name.pdf).
