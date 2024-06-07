import pandas as pd
import json
from datetime import date
from weasyprint import HTML

def get_trait_to_mean_from_raw_results(raw_results : pd.DataFrame, trait_info : pd.DataFrame) -> dict:
    
    """
    Extracts means for traits from raw results and maps them to trait IDs.
    
    Parameters:
    - raw_results (pd.DataFrame): DataFrame containing raw data, with means in a specific column format.
    - trait_info (pd.DataFrame): DataFrame with 'trait_id' column, mapping traits to IDs.

    Returns:
    - dict: A dictionary mapping trait IDs to their corresponding means.
    """

    mu_sigma_list = list(raw_results.iloc[:,1][11:])
    new_list = mu_sigma_list[6:36] + mu_sigma_list[:5]
    mu_list = [float(pair.split(' - ')[0]) for pair in new_list]
    traits_list = list(trait_info['trait_id'])
    
    trait_to_mean = dict(zip(traits_list,mu_list))
    
    return trait_to_mean

def get_trait_to_std_from_raw_results(raw_results : pd.DataFrame, trait_info : pd.DataFrame) -> dict:
    
    """
    Extracts standard deviations for traits from raw results and maps them to trait IDs.
    
    Parameters:
    - raw_results (pd.DataFrame): DataFrame containing raw data, with standard deviations in a specific column format.
    - trait_info (pd.DataFrame): DataFrame with 'trait_id' column, mapping traits to IDs.

    Returns:
    - dict: A dictionary mapping trait IDs to their corresponding standard deviations.
    """

    mu_sigma_list = list(raw_results.iloc[:,1][11:])
    new_list = mu_sigma_list[6:36] + mu_sigma_list[:5]
    sigma_list = [float(pair.split(' - ')[1]) for pair in new_list]
    traits_list = list(trait_info['trait_id'])
    
    trait_to_std = dict(zip(traits_list,sigma_list))
    
    return trait_to_std

def format_results (df : pd.DataFrame, trait_info_df : pd.DataFrame, item_info_df : pd.DataFrame) -> pd.DataFrame :

    """
    Processes raw questionnaire data to produce a standardized and enriched DataFrame with personality traits scores,
    Sloan categories, and job preferences based on the Sloan personality model.

    The function performs the following steps:
    - Calculates mean and standard deviation for each trait based on raw results and trait information.
    - Transforms and formats the raw DataFrame to align with trait and item metadata.
    - Standardizes trait scores using calculated means and standard deviations.
    - Classifies traits into Sloan categories (e.g., 'Social' vs. 'Reserved') based on standardized scores.
    - Determines Sloan personality type (a combination of Sloan categories) for each individual.
    - Maps each Sloan personality type to a most favored job.
    - Processes names to remove capital letters.

    Parameters:
    - df (pd.DataFrame): Raw questionnaire results, including respondent IDs and their responses.
    - trait_info_df (pd.DataFrame): Information about the traits, including IDs and descriptions.
    - item_info_df (pd.DataFrame): Information about the questionnaire items, including IDs and associated traits.

    Returns:
    - pd.DataFrame: The enriched DataFrame with standardized scores, Sloan categories, Sloan types,
      most favored jobs based on Sloan types, and formatted names.
    """

    trait_to_mean = get_trait_to_mean_from_raw_results(df, trait_info_df)

    trait_to_std = get_trait_to_std_from_raw_results(df, trait_info_df)

    # Format QBG results
    df = df.T
    df.columns = df.iloc[0]
    df = df.iloc[2:]
    prenoms = df.index.tolist()
    df.insert(0, 'Prénom', prenoms)
    columns = df.columns.tolist()
    columns = [x for x in columns if str(x).lower() != 'nan']
    df = df[columns]
    df[columns[8:]] = df[columns[8:]].astype(float)
    df.columns.name = None
    df.reset_index(drop=True,inplace=True)

    # Import the tables with all the info about traits (facettes and domaines are both considered as traits) and items
    item_info_df = pd.read_csv('input_data/item_info.csv')
    trait_info_df = pd.read_csv('input_data/trait_info.csv')

    # Rename columns
    columns[8:13] = list(trait_info_df['trait_id'])[-5:]
    columns[13:43] = list(trait_info_df['trait_id'])[0:30]
    columns[43:253] = list(item_info_df['item_id'])
    df.columns = pd.Series(columns)

    # Create a unique id column for each invididual
    df['unique_id'] = df.index

    # Define the list of traits
    traits_list = list(trait_info_df['trait_id'])

    # Change the data types of traits, from object to float
    df[traits_list] = df[traits_list].astype(float)

    # Standardize the data using the stored means and std for each trait
    for trait_id in traits_list:
        mu = trait_to_mean[trait_id]
        sigma = trait_to_std[trait_id]
        df[trait_id] = ((df[trait_id] - mu) / sigma) * (0.5/3) + 0.5

    # Add Sloan Domain Categories to the scores table
    df['E_categ'] = df['E'].apply(lambda x: 'Social' if x > 0.5 else 'Reserved')
    df['N_categ'] = df['N'].apply(lambda x: 'Limbic' if x > 0.5 else 'Calm')
    df['C_categ'] = df['C'].apply(lambda x: 'Organized' if x > 0.5 else 'Unstructured')
    df['A_categ'] = df['A'].apply(lambda x: 'Accomodating' if x > 0.5 else 'Egocentric')
    df['O_categ'] = df['O'].apply(lambda x: 'Inquisitive' if x > 0.5 else 'Non-curious')

    # Add sloan types to the scores table
    df['sloan_type'] = ''
    for letter in ['E','N','C','A','O']:
        df['sloan_type'] += df[f"{letter}_categ"].apply(lambda x: x[0])

    # Build the dict which returns the most favoured job for each sloan type
    sloan_type_to_mf_job_dict = {}
    sloan_type_to_mf_job_dict[f'SCOAN'] = 'event planner'
    sloan_type_to_mf_job_dict[f'RLUEI'] = 'freelance writer'
    sloan_type_to_mf_job_dict[f'RLUAI'] = 'philosophy professor'
    sloan_type_to_mf_job_dict[f'RCOEN'] = 'banker'
    sloan_type_to_mf_job_dict[f'SCUAI'] = 'entertainer'
    sloan_type_to_mf_job_dict[f'SCOEN'] = 'business consultant'
    sloan_type_to_mf_job_dict[f'SLUEI'] = 'film director'
    sloan_type_to_mf_job_dict[f'RCOAN'] = 'pediactric nurse'
    sloan_type_to_mf_job_dict[f'RLOAN'] = 'home maker'
    sloan_type_to_mf_job_dict[f'RCOEI'] = 'engineer'
    sloan_type_to_mf_job_dict[f'SLOAN'] = 'kindergarten teacher'
    sloan_type_to_mf_job_dict[f'SLUAI'] = 'journalist'
    sloan_type_to_mf_job_dict[f'SCUAN'] = 'comedian'
    sloan_type_to_mf_job_dict[f'RCUEI'] = 'video game designer'
    sloan_type_to_mf_job_dict[f'RLOAI'] = 'scholar'
    sloan_type_to_mf_job_dict[f'SLUAN'] = 'hair stylist'
    sloan_type_to_mf_job_dict[f'RLOEN'] = 'secretary'
    sloan_type_to_mf_job_dict[f'SLUEN'] = 'model'
    sloan_type_to_mf_job_dict[f'SCOEI'] = 'businessman'
    sloan_type_to_mf_job_dict[f'RLUAN'] = 'florist'
    sloan_type_to_mf_job_dict[f'SCUEI'] = 'international spy'
    sloan_type_to_mf_job_dict[f'RLOEI'] = 'astronomer'
    sloan_type_to_mf_job_dict[f'SCUEN'] = 'entertainment industry'
    sloan_type_to_mf_job_dict[f'SCOAI'] = 'medical doctor'
    sloan_type_to_mf_job_dict[f'RCOAI'] = 'research scientist'
    sloan_type_to_mf_job_dict[f'RLUEN'] = 'digital artist'
    sloan_type_to_mf_job_dict[f'SLOEI'] = 'politician'
    sloan_type_to_mf_job_dict[f'SLOEN'] = 'fashion merchandiser'
    sloan_type_to_mf_job_dict[f'RCUAN'] = 'animal trainer'
    sloan_type_to_mf_job_dict[f'RCUAI'] = 'archeologist'
    sloan_type_to_mf_job_dict[f'RCUEN'] = 'race car driver'
    sloan_type_to_mf_job_dict[f'SLOAI'] = 'counselor'

    # Add the sloan_most_favored_job column into the main table
    df['sloan_most_favored_job'] = df['sloan_type'].apply(lambda x: sloan_type_to_mf_job_dict[x])

    # Remove capital letters from 'Prénom' and 'Nom'
    df['Prénom'] = df['Prénom'].str.lower()
    df['Nom'] = df['Nom'].str.lower()

    return df

def soft_skill_retreiver(client):
    seed = 3
    with open('input_data/system_prompt_1.txt', 'r') as file:
        system_prompt = file.read()
    with open('input_data/job_description.txt', 'r') as file:
        job_description = file.read()

    system_message = {
        "role" : "system",
        "content" : system_prompt
    }
    user_message = {
        "role" : "user",
        "content" : "Fiche de poste :" + job_description
    }

    # Create the chat completion
    completion = client.chat.completions.create(
      model="gpt-4",
      seed=seed,
      temperature=0.1,
      messages=[system_message,user_message]
    )

    with open("output_data/llm_step_1_output.txt", 'w') as file:
        file.write(completion.choices[0].message.content)

    output_dict = json.loads(completion.choices[0].message.content)

    with open("output_data/llm_step_1_output.json", 'w') as file:
        json.dump(output_dict, file)


def soft_skill_converter(client):

    seed = 3
    with open('input_data/system_prompt_2.txt', 'r') as file:
        system_prompt = file.read()
    with open('output_data/llm_step_1_output.txt', 'r') as file:
        job_description = file.read()

    system_message = {
        "role" : "system",
        "content" : system_prompt
    }
    user_message = {
        "role" : "user",
        "content" : job_description
    }

    # Create the chat completion
    completion = client.chat.completions.create(
      model="gpt-4",
      seed=seed,
      temperature=0.1,
      messages=[system_message,user_message]
    )

    with open("output_data/llm_step_2_output.txt", 'w') as file:
        file.write(completion.choices[0].message.content)


def get_job_description_soft_skills(client) -> dict:

    """
    """
    soft_skill_retreiver(client)
    soft_skill_converter(client)

    # Get the response as a string
    with open("output_data/llm_step_2_output.txt", 'r') as file:
        job_description_soft_skills = file.read()

    # Store the output
    with open("output_data/llm_output.txt", "w") as file:
        file.write(job_description_soft_skills)

    if job_description_soft_skills != None:

        # Convert the string to a Python dictionary
        job_description_soft_skills_dict = json.loads(job_description_soft_skills)
        job_description_soft_skills = job_description_soft_skills_dict["soft_skills"]

    else:
        job_description_soft_skills = []

    return job_description_soft_skills

def get_job_description_specific_traits(job_description_soft_skills:dict, skill_to_traits_dict: dict) -> list:

    """
    Maps soft skills extracted from a job description to specific personality traits.

    This function takes a dictionary of soft skills identified in a job description and a mapping dictionary
    that relates soft skills to specific personality traits. It then creates a list of unique personality traits
    that are associated with the soft skills required for the job.

    Parameters:
    - job_description_soft_skills (dict): A dictionary containing soft skills as keys, extracted from a job description.
    - skill_to_traits_dict (dict): A dictionary mapping each soft skill to one or more corresponding personality traits.

    Returns:
    - list: A list of unique personality traits associated with the required soft skills for the job. If no soft skills
      are found or if the soft skills do not map to any traits in the provided dictionary, an empty list is returned.

    Note:
    - The function only includes unique traits, avoiding duplicates even if multiple soft skills map to the same trait.
    """

    job_description_specific_traits = []
    if len(job_description_soft_skills) != 0:
        for soft_skill in job_description_soft_skills:
            if soft_skill in skill_to_traits_dict.keys():
                for trait in skill_to_traits_dict[soft_skill]:
                    if trait not in job_description_specific_traits:
                        job_description_specific_traits.append(trait)

    return job_description_specific_traits


def trait_value_to_modality(trait_value: float, trait_gender: str):

    '''
    Converts a numerical trait value into a qualitative description based on the value range and gender.

    This function is designed to provide qualitative assessments of personality traits or other characteristics
    by translating a numerical score (ranging from 0 to 1) into a gender-specific qualitative description.
    It can be particularly useful in generating readable personality reports where traits are assessed on a continuum
    but need to be communicated in a more relatable, qualitative manner.

    Parameters:
    - trait_value (float): The numerical value of the trait, which should be between 0 and 1 (inclusive).
    - trait_gender (str): A string indicating the gender ('m' for masculine, 'f' for feminine) to select the appropriate qualitative descriptor.

    Returns:
    - str: A gender-appropriate, qualitative description of the trait value. It returns one of eight different
      descriptions ('très bas', 'très basse', 'bas', 'basse', 'élevé', 'élevée', 'très élevé', 'très élevée')
      based on the combination of trait value and gender. If the input is invalid, it returns 'Invalid input'.

    Examples:
    >>> trait_value_to_modality(0.2, 'm')
    'très bas'
    >>> trait_value_to_modality(0.65, 'f')
    'élevée'
    '''

    if 0 <= trait_value <= 0.3:
        if trait_gender == 'm':
            return 'très bas'
        else:
            return 'très basse'
    elif 0.3 < trait_value <= 0.5:
        if trait_gender == 'm':
            return 'bas'
        else:
            return 'basse'
    elif 0.5 < trait_value <= 0.7:
        if trait_gender == 'm':
            return 'élevé'
        else:
            return 'élevée'
    elif 0.7 < trait_value <= 1:
        if trait_gender == 'm':
            return 'très élevé'
        else:
            return 'très élevée'
    else:
        return 'Invalid input'

def get_trait_name_to_modality_for_candidate(candidate_dict, trait_info_df):

    """
    Converts candidate trait information into a dictionary mapping trait names to their qualitative levels.

    This function takes a dictionary containing a candidate's trait values (candidate_dict) and a DataFrame
    (trait_info_df) which includes information about each trait such as its ID, grammatical gender, and name. It processes
    these inputs to produce a dictionary where the keys are trait names and the values are the qualitative levels
    of these traits for the given candidate, based on their numerical values and corresponding genders.

    Parameters:
    - candidate_dict (dict): A dictionary where keys are trait IDs and values are the candidate's scores for those traits.
    - trait_info_df (pandas.DataFrame): A DataFrame containing at least three columns: 'trait_id', 'trait_gender', and 'intitulé_trait'
      which represent the trait's ID, the gender associated with the trait, and the trait's name, respectively.

    Returns:
    - dict: A dictionary mapping trait names (str) to the qualitative levels (str or numeric, depending on the implementation of
      `trait_value_to_modality`) of those traits for the specified candidate.
    """

    # Define the list of traits
    traits_id_list = list(trait_info_df['trait_id'])

    # Create the dictionary : trait_id -> trait_gender
    trait_id_to_trait_gender = dict(zip(trait_info_df['trait_id'], trait_info_df['trait_gender']))

    # Create the dictionary : trait_id -> trait_name
    trait_id_to_trait_name = dict(zip(trait_info_df['trait_id'], trait_info_df['intitulé_trait']))

    # Build the dictionary : trait_name  --> qualitative level of the candidate for the trait
    trait_name_to_modality_for_candidate = {}
    for trait_id in traits_id_list:
        trait_value = candidate_dict[trait_id]
        trait_gender = trait_id_to_trait_gender[trait_id]
        trait_name = trait_id_to_trait_name[trait_id]
        trait_name_to_modality_for_candidate[trait_name] = trait_value_to_modality(trait_value, trait_gender)
    
    return trait_name_to_modality_for_candidate

def get_trait_name_to_ideal_modality_for_job(job_description_specific_traits):

    """
    Converts a list of trait_modality pairs into a dictionary.

    Each item in the input list is a string containing exactly two words: a trait name
    and its ideal modality, separated by a space. This function splits these items
    and stores them in a dictionary where the trait names are keys and the modalities
    are their corresponding values.

    Parameters:
    - job_description_specific_traits (list of str): List of "trait modality" pairs.

    Returns:
    - dict: A dictionary mapping traits to their ideal modalities.

    Example:
    >>> get_trait_name_to_ideal_modality_for_job(["Agréabilité élevée", "Tristesse basse"])
    {'Agréabilité': 'élevée', 'Tristesse': 'basse'}
    """

    trait_name_to_ideal_modality_for_job = {}
    for item in job_description_specific_traits:
        trait, modality = item.rsplit(' ', 1)
        trait_name_to_ideal_modality_for_job[trait] = modality

    return trait_name_to_ideal_modality_for_job

def get_candidate_top_values(candidate_id : int, df : pd.DataFrame, trait_info_df : pd.DataFrame):

    """
    Identifies a candidate's most defining personality traits based on their scores.

    This function analyzes a candidate's scores across various personality traits, identifying those traits that show
    significant deviation from a baseline score of 0.5. Traits with scores below 0.3 or above 0.7 are considered
    particularly defining. The function first prioritizes domain-level traits ('O', 'C', 'E', 'A', 'N') based on
    their deviation from 0.5, and then selects facet-level traits that are not redundant with the eventually already
    selected domains. The selection process aims to identify a minimum of five and maximum of six traits that best describe
    the candidate, with their names returned based on mappings provided in `trait_info_df`.

    Parameters:
    - candidate_id (int): The unique identifier for the candidate.
    - df (pd.DataFrame): The DataFrame containing candidates' scores across various traits.
    - trait_info_df (pd.DataFrame): A DataFrame mapping trait IDs to their descriptive names.

    Returns:
    - list of str: The names of five to six "top personality traits for the candidate".

    """

    # Create the dictionary : trait_id -> trait_name
    trait_id_to_trait_name = dict(zip(trait_info_df['trait_id'], trait_info_df['intitulé_trait']))

    # Get the list of traits
    traits_list = list(trait_info_df['trait_id'])

    # Build the df with all the sorted traits of the candidate, based on their distance to 0.5, with descending order
    candidate_df = df.loc[df['unique_id'] == candidate_id, traits_list]
    candidate_df = candidate_df.reindex(candidate_df.iloc[0].apply(lambda x: abs(0.5 - x)).sort_values().index, axis=1)
    candidate_df = candidate_df.iloc[:, ::-1]

    # Initialize the list of top values
    candidate_top_values = []

    # Add the domains superior to 0.7 or inferior to 0.3
    domains_list = ['O','C','E','A','N']
    for trait_id in list(candidate_df.columns):
        if trait_id in domains_list:
            if candidate_df[trait_id].values[0] < 0.3 or candidate_df[trait_id].values[0] > 0.7 :
                candidate_top_values.append(trait_id)

    # Add the strongest facettes, if they do not belong to the domains already added
    for trait_id in list(candidate_df.columns):
        if trait_id not in domains_list:
            if f"{trait_id}"[:-1] not in candidate_top_values:
                candidate_top_values.append(trait_id)

    # Keep only 6 traits if the 6th is superior to 0.7 or inferior to 0.3, else keep only 5 traits
    if candidate_df[candidate_top_values[5]].values[0] < 0.3 or candidate_df[candidate_top_values[5]].values[0] > 0.7 :
        candidate_top_values = candidate_top_values[:6]
    else:
        candidate_top_values = candidate_top_values[:5]

    # Convert id to names
    candidate_top_values = [trait_id_to_trait_name[i] for i in candidate_top_values]

    return candidate_top_values

def get_missing_traits(selected_traits : list,
                       job_to_traits : dict,
                       job_name : str):
    
    """
    Identifies traits required for a job that are not present in a given list of selected traits.

    This function compares a predefined list of traits necessary for a specific job (as defined in the `job_to_traits` dictionary)
    against a list of already selected traits, based on the analysis of the job description.
    It returns a list of traits that are considered necessary for the job but are missing
    from the selected traits.

    Parameters:
    - selected_traits (list): A list of trait names that have been selected or identified as important for the job, 
    based only on the text of the job description.
    - job_to_traits (dict): A dictionary mapping job names (str) to lists of traits (list of str) considered important for those jobs.
    - job_name (str): The name of the job for which the missing traits are to be identified.

    Returns:
    - list: A list of trait names required for the job but not included in the `selected_traits` list.

    Example:
    >>> job_to_traits = {"Engineer": ["creativity", "problem-solving", "teamwork"]}
    >>> selected_traits = ["creativity", "teamwork"]
    >>> get_missing_traits(selected_traits, job_to_traits, "Engineer")
    ['problem-solving']
    """

    missing_traits = [trait for trait in job_to_traits[job_name] if trait not in selected_traits]

    return missing_traits

def remove_modality(trait : str) -> str:

    """
    Removes the modality from a trait description.

    This function takes a trait description consisting of one or more words followed by a modality
    indicator (e.g., "Agréabilité élevée") and removes the modality part, returning just the trait name.

    Parameters:
    - trait (str): A string representing a trait followed by its modality, separated by spaces.

    Returns:
    - str: The trait name without the modality part.

    Example:
    >>> remove_modality("Agréabilité élevée")
    'Agréabilité'
    """

    trait = " ".join(trait.split()[:-1])
    return trait

def add_manager_traits(selected_traits : list, missing_traits : list, job_to_traits : dict) -> list:

    """
    Updates the list of missing traits for a job by adding missing manager-specific traits.

    This function takes lists of selected and missing traits for a job, along with a dictionary mapping
    jobs to their required traits, and identifies any manager-specific traits not already included in the
    total list of traits (selected + missing). It then adds these missing manager traits to the original
    list of missing traits and returns the updated list.

    Both selected and missing traits can include modalities (e.g., "creativity high"). This function removes
    these modalities to compare the trait names directly. It ensures that all necessary manager traits,
    as defined in `job_to_traits` under "manager", are considered, even if they weren't initially selected
    or identified as missing for the specific job.

    Parameters:
    - selected_traits (list): A list of trait names (with or without modalities) already selected.
    - missing_traits (list): A list of trait names (with or without modalities) identified as missing for the job.
    - job_to_traits (dict): A dictionary mapping job titles to lists of their required traits (with or without modalities).

    Returns:
    - list: An updated list of missing traits including any missing manager-specific traits.

    Example:
    >>> job_to_traits = {"manager": ["Extraversion", "Franc-parler élevé", "Leadership élevé", "Exploration basse"]}
    >>> selected_traits = ["Franc-parler élevé"]
    >>> missing_traits = ["Leadership élevé",]
    >>> add_manager_traits(selected_traits, missing_traits, job_to_traits)
    ["Extraversion", "Franc-parler élevé", "Leadership élevé", "Exploration basse"]
    """

    selected_traits_no_modality = [remove_modality(trait) for trait in selected_traits]
    missing_traits_no_modality = [remove_modality(trait) for trait in missing_traits]
    total_traits = selected_traits_no_modality + missing_traits_no_modality
    new_missing_traits = missing_traits + [trait for trait in job_to_traits["manager"] if remove_modality(trait) not in total_traits]
    
    return new_missing_traits

def select_traits_and_create_report(job_name : str, with_job_description : bool, manager_job : bool, client):

    """
    Selects traits and creates reports based on a job description and job type.

    This function processes a job description to identify relevant soft skills and specific traits, then generates
    reports based on these traits. It can operate in two modes: one that includes processing a job description to
    extract soft skills and their corresponding traits, and another that skips this step. The function supports an
    additional feature to adjust the selection process if the job is a managerial position.

    For job descriptions, the function reads and processes the text to extract soft skills using an external AI service
    (represented by the `client` parameter). It maps these soft skills to specific traits, generates various reports,
    and saves them to files. For jobs without a description or when skipping this step, it proceeds directly to report
    generation based on pre-existing data or a lack thereof.

    Parameters:
    - job_name (str): The name of the job for which traits and reports are being generated.
    - with_job_description (bool): A flag indicating whether to process a job description to extract skills and traits.
    - manager_job (bool): Indicates if the job is a managerial position, which affects the trait selection process.
    - client: OpenAI client used for processing job descriptions to extract soft skills.

    The function performs file operations to read job descriptions, soft skills, and trait mappings, and writes out
    the results of the traits recognition process and selected traits for the job. It leverages external services
    for text processing and requires various input and output data files.

    """
        
    # Set seed for reproducibility
    seed = 43

    with open('input_data/job_to_traits.json','r') as file:
        job_to_traits = json.load(file)

    ### openai part

    if with_job_description == True:
        job_soft_skills = get_job_description_soft_skills(client)

        with open('output_data/job_soft_skills.json','w') as file:
            json.dump(job_soft_skills, file)

        with open('input_data/skill_to_traits.json', 'r') as file:
            skill_to_traits_dict = json.load(file)

        job_description_specific_traits = get_job_description_specific_traits(job_soft_skills, skill_to_traits_dict)

        with open('output_data/job_description_specific_traits.json', 'w') as file:
            file.write(json.dumps(job_description_specific_traits))

        with open('output_data/selected_traits.json', 'w') as file:
            file.write(json.dumps(job_description_specific_traits))
    
    else:
        job_description_specific_traits = []

        with open('output_data/job_description_specific_traits.json', 'w') as file:
            file.write(json.dumps(job_description_specific_traits))

        with open('output_data/selected_traits.json', 'w') as file:
            file.write(json.dumps(job_description_specific_traits))


    get_traits_recognition_report(job_description_specific_traits, job_to_traits, job_name, manager_job)

def get_traits_recognition_report(selected_traits : list,
                                  job_to_traits : dict,
                                  job_name : str,
                                  manager_job : bool) -> str:
    
    """
    Generates an HTML report detailing the identification of relevant personality traits for a specific job.

    This function creates a report that lists the personality traits selected based on a job description and
    identifies any additional traits recommended for the job, especially if it's a managerial position. The report
    differentiates between traits that were directly identified from the job description and those that are deemed
    necessary but were missing from the initial selection. The function handles jobs with no directly selected traits
    by suggesting traits based on the job role and whether it includes managerial responsibilities.

    The generated HTML report includes styling for readability and is saved to a specified location. The function
    also outputs to the console the path to the saved report.

    Parameters:
    - selected_traits (list): A list of personality trait names that have been identified as relevant to the job.
    - job_to_traits (dict): A dictionary mapping job titles to lists of required personality traits.
    - job_name (str): The name of the job for which the report is being generated.
    - manager_job (bool): A boolean flag indicating whether the job is a managerial position, affecting the trait recommendations.

    Returns:
    - str: The path to the saved HTML report file.

    The function also writes a JSON file with missing traits (considering the managerial adjustments if applicable)
    and prints the file path of the generated HTML report.
    """
    
    file_content = f"""
    <!DOCTYPE html>
    <html>
    <meta charset="UTF-8">
    <head>
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel='stylesheet'>
        <style>
            html, body {{
                font-family: 'Montserrat', sans-serif;
                background-color: #101f56;
                color: white;
                min-height: 100%;
            }}
        </style>
    </head>
    <body>
        <h1>Identification des traits de personnalité pertinents :</h1>
    </body>
    </html>
    """    
    
    if len(selected_traits) == 0:
        file_content += "<p>A partir de la fiche de poste fournie, je n'ai pas identifié de trait de personnalité correspondant.</p>"

        if job_name in job_to_traits:
            missing_traits = get_missing_traits(selected_traits, job_to_traits, job_name)
            if manager_job:
                missing_traits = add_manager_traits(selected_traits, missing_traits, job_to_traits)
            with open("output_data/missing_traits.json",'w') as file:
                json.dump(missing_traits, file)
            if missing_traits:
                missing_traits_list = "<li>" + "</li><li>".join(missing_traits) + "</li>"
                if len(missing_traits) == 1:
                    file_content += f"<p>Pour un poste de {job_name}, je suggère de prendre en compte le trait de personnalité suivant : {missing_traits_list}</p>"
                    file_content += f"<p>Souhaitez-vous l'inclure dans l'analyse ?<p>"

                elif len(missing_traits) > 1:
                    file_content += f"<p>Pour un poste de {job_name}, je suggère de prendre également en compte les traits suivants :</p><ul>{missing_traits_list}</ul>"
                    file_content += f"<p>Souhaitez-vous les inclure dans l'analyse ?<p>"
        
        else:
            missing_traits = []
            with open("output_data/missing_traits.json",'w') as file:
                json.dump(missing_traits, file)

    elif len(selected_traits) > 0:
        selected_traits_list = "<li>" + "</li><li>".join(selected_traits) + "</li>"
        file_content += f"<p>A partir de la fiche de poste fournie, j'ai identifié les traits de personnalité suivants comme importants :</p><ul>{selected_traits_list}</ul>"

        if job_name in job_to_traits:
            missing_traits = get_missing_traits(selected_traits, job_to_traits, job_name)
            if manager_job:
                missing_traits = add_manager_traits(selected_traits, missing_traits, job_to_traits)
            with open("output_data/missing_traits.json",'w') as file:
                json.dump(missing_traits, file)
            if missing_traits:
                missing_traits_list = "<li>" + "</li><li>".join(missing_traits) + "</li>"
                if len(missing_traits) == 1:
                    file_content += f"<p>Pour un poste de {job_name}, je suggère de prendre également en compte le trait de personnalité suivant : {missing_traits[0]}.</p>"
                    file_content += f"<p>Souhaitez-vous l'inclure dans l'analyse ?</p>"

                elif len(missing_traits) > 1:
                    file_content += f"<p>Pour un poste de {job_name}, je suggère de prendre également en compte les traits de personnalité suivants :</p><ul>{missing_traits_list}</ul>"
                    file_content += f"<p>Souhaitez-vous les inclure dans l'analyse ?</p>"
        else:
            missing_traits = []
            with open("output_data/missing_traits.json",'w') as file:
                json.dump(missing_traits, file)

    html_file_path = f"output_data/traits_recognition_report.html"
    with open(html_file_path, 'w', encoding='utf-8') as file:
        file.write(file_content)
    print(f"Report saved to {html_file_path}")

def add_missing_traits() -> None:

    """
    Adds missing traits to the list of selected traits for a job.

    This function reads a list of missing traits from a JSON file and a list of traits that have already been
    selected based on a job description from another JSON file. It combines these two lists, updating the list
    of selected traits to include the previously identified missing traits. The updated list is then saved to a
    JSON file, effectively merging the missing traits into the list of selected traits for further analysis or
    reporting.

    No parameters are required for this function, and it does not return any value. It operates solely through
    reading from and writing to files specified within its implementation.

    Files involved:
    - Reads missing traits from 'output_data/missing_traits.json'.
    - Reads selected traits from 'output_data/job_description_specific_traits.json'.
    - Writes the updated list of selected traits to 'output_data/selected_traits.json'.
    """

    with open('output_data/missing_traits.json','r') as file:
        missing_traits = json.load(file)
    with open('output_data/job_description_specific_traits.json', 'r') as file:
        selected_traits = json.load(file)

    selected_traits.extend(missing_traits)

    with open('output_data/selected_traits.json','w') as file:
        json.dump(selected_traits, file)


def get_document(candidate_first_name : str,
                candidate_name : str,
                job_name : str,
                job_description_specific_traits : list,
                trait_name_to_modality_for_candidate : dict,
                trait_name_to_ideal_modality_for_job : dict,
                candidate_top_values : list,
                trait_info_df : pd.DataFrame,
                candidate_id : int,
                sloan_type : str):
    
    """
    Generates a personalized HTML report evaluating a candidate's personality traits against job requirements.

    This report assesses the alignment between a candidate's traits (and their modalities) with the ideal traits (and
    modalities) required for a specific job. It visually highlights traits that are a match or mismatch with the job's
    requirements and includes personalized questions or comments based on the trait analysis. The report focuses on
    the specific traits derived from the job description and the candidate's top values, offering insights into the
    candidate's potential fit for the job and areas for further inquiry.

    Parameters:
    - candidate_first_name (str): The candidate's first name.
    - candidate_name (str): The candidate's last name.
    - job_name (str): The name of the job.
    - job_description_specific_traits (list): A list of traits specified in the job description.
    - trait_name_to_modality_for_candidate (dict): A dictionary mapping trait names to their modalities for the candidate.
    - trait_name_to_ideal_modality_for_job (dict): A dictionary mapping trait names to their ideal modalities for the job.
    - candidate_top_values (list): A list of the candidate's top trait values.
    - trait_info_df (pd.DataFrame): A DataFrame containing additional information about each trait.
    - candidate_id (int): The candidate's unique identifier.
    - sloan_type (str): The Sloan personality type of the candidate.

    The function creates an HTML file with the report content, including an adequacy index and tailored questions
    based on the trait analysis. The report is saved with a filename pattern that incorporates the candidate's name,
    Sloan type, and ID, facilitating easy identification.

    Outputs:
    - The path to the saved HTML report is printed to the console.
    """

    high_modalities_list = ["élevé", "élevée", "très élevé", "très élevée"]
    low_modalities_list = ["bas", "basse", "très bas", "très basse"]
    
    file_content = f"""
    <!DOCTYPE html>
    <html>
    <meta charset="UTF-8">
    <head>
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel='stylesheet'>
        <style>
            html, body {{
                font-family: 'Montserrat', sans-serif;
                background-color: #101f56;
                color: white;
                min-height: 100%;
            }}
            .legend {{
                font-size: 16px;
                margin-top: 10px;
                float: right;
            }}
            .full-height {{
            min-height: 110vh; /* Slightly more than 100% of the viewport height to ensure full coverage */
            }}
            .color-box {{
                display: inline-block;
                width: 20px;
                height: 20px;
                margin-right: 5px;
                vertical-align: middle;
            }}
            .adequation {{
                background-color: #55e975;
            }}
            .inadequation {{
                background-color: #ff9548;
            }}

            .content {{
                margin-left: 30px;
                font-size: 18px;
            }}
            .page-break {{
                page-break-before: always;
            }}
            .h1 {{
                font-size : 30px;

            }}
        </style>
    </head>
    <body>
        <h1>Rapport d'adéquation entre la personnalité de {candidate_first_name} et le poste de {job_name} :</h1>
        <div class="legend">
            <span class="color-box adequation"></span>Adéquation |
            <span class="color-box inadequation"></span>Inadéquation
        </div>
        <br>
    </body>
    </html>
    """

    # Job description specific content
    if len(job_description_specific_traits) != 0:

        fit_count = 0
        file_content += "<h2>Adéquation avec le poste :</h2>"

        for trait_with_level in job_description_specific_traits:
            trait_name = trait_with_level.rsplit(' ',1)[0]
            candidate_trait_modality = trait_name_to_modality_for_candidate[trait_name]
            job_ideal_modality = trait_name_to_ideal_modality_for_job[trait_name]

            if candidate_trait_modality in high_modalities_list:
                if job_ideal_modality in high_modalities_list:
                    fit_value = 'match'
                else:
                    fit_value = 'mismatch'
            if candidate_trait_modality in low_modalities_list:
                if job_ideal_modality in low_modalities_list:
                    fit_value = 'match'
                else:
                    fit_value = 'mismatch'
            
            if fit_value == 'match':
                trait_color = '#55e975'
                fit_count +=1
            else:
                trait_color = '#ff9548'

            file_content += f"<p class='content'><span style='color:{trait_color};'><strong>{trait_name} {candidate_trait_modality}</strong></span></p>"


            if candidate_trait_modality in low_modalities_list and job_ideal_modality in low_modalities_list:
                question = (f"{trait_info_df.loc[trait_info_df['intitulé_trait'] == trait_name,'bas_question_potentialité_1'].values[0]}\n")
            if candidate_trait_modality in high_modalities_list and job_ideal_modality in high_modalities_list:
                question = (f"{trait_info_df.loc[trait_info_df['intitulé_trait'] == trait_name,'élevé_question_potentialité_1'].values[0]}\n")
            if candidate_trait_modality in low_modalities_list and job_ideal_modality in high_modalities_list:
                question = (f"{trait_info_df.loc[trait_info_df['intitulé_trait'] == trait_name,'bas_question_risque_1'].values[0]}\n")
            if candidate_trait_modality in high_modalities_list and job_ideal_modality in low_modalities_list:
                question = (f"{trait_info_df.loc[trait_info_df['intitulé_trait'] == trait_name,'élevé_question_risque_1'].values[0]}\n")
            
            file_content += f"<p class='content'>&rarr; {question}</p>"

        fit_count /= len(job_description_specific_traits)
        fit_percentage = int(fit_count*100)
        if fit_percentage >= 50:
            fit_color = '#55e975'
        else:
            fit_color = '#ff9548'

        #file_content += f"<p class='content'><b>Indice d'adéquation : </b><span style='color:{fit_color};'>{fit_percentage}%</span></p>"
    
    # Candidate strong traits specific content
    for trait_name in candidate_top_values:
        if trait_name in job_description_specific_traits:
            candidate_top_values.remove(trait_name)

    if len(candidate_top_values) != 0:
        file_content += f'''<h2>Partie spécifique aux traits marqués de {candidate_first_name} :</h2>'''
        

        for trait_name in candidate_top_values:
            file_content += f"<p class='content'><b>{trait_name} {trait_name_to_modality_for_candidate[trait_name]}</b></p>"
            if trait_name_to_modality_for_candidate[trait_name] in low_modalities_list:
                trait_global_modality = 'bas'
            elif trait_name_to_modality_for_candidate[trait_name] in high_modalities_list:
                trait_global_modality = 'élevé'

            question = f"{trait_info_df.loc[trait_info_df['intitulé_trait'] == trait_name,f'{trait_global_modality}_question_risque_1'].values[0]}\n"
            file_content += f"<p class='content'>&rarr; {question}</p>"

    html_file_path = f"output_data/candidates_reports/{candidate_first_name}_{candidate_name}_{sloan_type}_{candidate_id}_report.html"
    with open(html_file_path, 'w', encoding='utf-8') as file:
        file.write(file_content)
    print(f"Report saved to {html_file_path}")

def jobmatch_data(job_description_specific_traits : list,
                  trait_name_to_modality_for_candidate : dict,
                  trait_name_to_ideal_modality_for_job : dict,
                  trait_info_df : pd.DataFrame):
                
    doc_data = [{"trait_modality": None, "fit": None, "question": None} for _ in range(len(job_description_specific_traits))]

    high_modalities_list = ["élevé", "élevée", "très élevé", "très élevée"]
    low_modalities_list = ["bas", "basse", "très bas", "très basse"]

    # Job description specific content
    if len(job_description_specific_traits) != 0:

        for i, trait_with_level in enumerate(job_description_specific_traits):
            trait_name = trait_with_level.rsplit(' ',1)[0]
            candidate_trait_modality = trait_name_to_modality_for_candidate[trait_name]
            job_ideal_modality = trait_name_to_ideal_modality_for_job[trait_name]

            if candidate_trait_modality in high_modalities_list:
                if job_ideal_modality in high_modalities_list:
                    fit = True
                else:
                    fit = False
            if candidate_trait_modality in low_modalities_list:
                if job_ideal_modality in low_modalities_list:
                    fit = True
                else:
                    fit = False

            doc_data[i]["fit"] = fit

            doc_data[i]["trait_modality"] = trait_name + " " + candidate_trait_modality

            if candidate_trait_modality in low_modalities_list and job_ideal_modality in low_modalities_list:
                question = (f"{trait_info_df.loc[trait_info_df['intitulé_trait'] == trait_name,'bas_question_potentialité_1'].values[0]}\n")
            if candidate_trait_modality in high_modalities_list and job_ideal_modality in high_modalities_list:
                question = (f"{trait_info_df.loc[trait_info_df['intitulé_trait'] == trait_name,'élevé_question_potentialité_1'].values[0]}\n")
            if candidate_trait_modality in low_modalities_list and job_ideal_modality in high_modalities_list:
                question = (f"{trait_info_df.loc[trait_info_df['intitulé_trait'] == trait_name,'bas_question_risque_1'].values[0]}\n")
            if candidate_trait_modality in high_modalities_list and job_ideal_modality in low_modalities_list:
                question = (f"{trait_info_df.loc[trait_info_df['intitulé_trait'] == trait_name,'élevé_question_risque_1'].values[0]}\n")
            
            doc_data[i]["question"] = question
    
    with open("output_data/jobmatch_data.json", "w") as file:
        json.dump(doc_data, file)

def get_infos(first_name : str, last_name : str, job_name : str):
    infos = {"first_name": first_name, "last_name": last_name, "job_name": job_name}

    with open("output_data/infos.json", "w") as file:
        json.dump(infos, file)

def strong_traits_data(job_description_specific_traits : list,
                    trait_name_to_modality_for_candidate : dict,
                    candidate_top_values : list,
                    trait_info_df : pd.DataFrame):


    for trait_name in candidate_top_values:
        if trait_name in job_description_specific_traits:
            candidate_top_values.remove(trait_name)

    traits_data = [{"trait_modality": None, "question": None} for _ in range(len(candidate_top_values))]

    high_modalities_list = ["élevé", "élevée", "très élevé", "très élevée"]
    low_modalities_list = ["bas", "basse", "très bas", "très basse"]

    if len(candidate_top_values) != 0:

        for i, trait_name in enumerate(candidate_top_values):
            traits_data[i]["trait_modality"] = trait_name + " " + trait_name_to_modality_for_candidate[trait_name]

            if trait_name_to_modality_for_candidate[trait_name] in low_modalities_list:
                trait_global_modality = 'bas'
            elif trait_name_to_modality_for_candidate[trait_name] in high_modalities_list:
                trait_global_modality = 'élevé'

            traits_data[i]["question"] = trait_info_df.loc[trait_info_df['intitulé_trait'] == trait_name,f'{trait_global_modality}_question_risque_1'].values[0]

    with open("output_data/traits_data.json", "w") as file:
        json.dump(traits_data, file)

def create_report(job_name : str, candidate_mail : str):

    """
    Generates a detailed report for a candidate based on their assessment results and job requirements.

    This function orchestrates the creation of a personalized report for a candidate by:
    - Reading assessment results and trait information from CSV files.
    - Formatting these results to match the required data structure.
    - Retrieving the list of selected traits for the job from a JSON file.
    - Mapping each selected trait to its ideal modality for the job.
    - Extracting the candidate's information and assessment scores based on their email.
    - Identifying the candidate's top traits and their modalities.
    - Calling a function to generate a detailed HTML report that assesses the fit between the candidate's
      personality traits and the job requirements, including additional considerations based on the candidate's
      most prominent traits.

    Parameters:
    - job_name (str): The name of the job for which the report is being created.
    - candidate_mail (str): The email address of the candidate, used to identify their assessment results.

    The function integrates various data processing steps and leverages helper functions to generate a comprehensive
    report. It assumes the existence of predefined data files (`results.csv`, `trait_info.csv`, `item_info.csv`, and
    `selected_traits.json`) and the availability of specific functions for processing this data (`format_results`,
    `get_trait_name_to_ideal_modality_for_job`, `get_trait_name_to_modality_for_candidate`, and `get_candidate_top_values`).
    Finally, it calls `get_document` to generate and save the HTML report, using the candidate's first name, last name,
    Sloan type, and unique ID to name the output file.
    """

    # Import the tables with scores (df) all the info about traits (facettes and domaines are both considered as traits) and items
    df = pd.read_csv('input_data/results.csv')
    trait_info_df = pd.read_csv('input_data/trait_info.csv')
    item_info_df = pd.read_csv('input_data/item_info.csv')

    # Format QBG results
    df = format_results(df, trait_info_df, item_info_df)

    with open('output_data/selected_traits.json','r') as file:
        selected_traits = json.load(file)

    selected_traits = [trait.replace('\u2019', "'") for trait in selected_traits]

    trait_name_to_ideal_modality_for_job = get_trait_name_to_ideal_modality_for_job(selected_traits)

    candidate_df = df.loc[df['Email'] == candidate_mail]

    candidate_dict = candidate_df.iloc[0].to_dict()

    first_name = candidate_dict['Prénom']
    name = candidate_dict['Nom']
    sloan_type = candidate_dict['sloan_most_favored_job']
    candidate_id = candidate_dict['unique_id']

    trait_name_to_modality_for_candidate = get_trait_name_to_modality_for_candidate(candidate_dict,trait_info_df)

    candidate_top_values = get_candidate_top_values(candidate_id, df, trait_info_df)

    get_document(first_name,
                    name,
                    job_name,
                    selected_traits,
                    trait_name_to_modality_for_candidate,
                    trait_name_to_ideal_modality_for_job,
                    candidate_top_values,
                    trait_info_df,
                    candidate_id,
                    sloan_type)
    
def get_report_data(job_name : str, candidate_mail : str):

    # Import the tables with scores (df) all the info about traits (facettes and domaines are both considered as traits) and items
    df = pd.read_csv('input_data/results.csv')
    trait_info_df = pd.read_csv('input_data/trait_info.csv')
    item_info_df = pd.read_csv('input_data/item_info.csv')

    # Format QBG results
    df = format_results(df, trait_info_df, item_info_df)

    with open('output_data/selected_traits.json','r') as file:
        selected_traits = json.load(file)

    selected_traits = [trait.replace('\u2019', "'") for trait in selected_traits]

    trait_name_to_ideal_modality_for_job = get_trait_name_to_ideal_modality_for_job(selected_traits)

    candidate_df = df.loc[df['Email'] == candidate_mail]

    candidate_dict = candidate_df.iloc[0].to_dict()

    first_name = candidate_dict['Prénom']
    last_name = candidate_dict['Nom']
    candidate_id = candidate_dict['unique_id']

    trait_name_to_modality_for_candidate = get_trait_name_to_modality_for_candidate(candidate_dict,trait_info_df)

    candidate_top_values = get_candidate_top_values(candidate_id, df, trait_info_df)


    jobmatch_data(selected_traits,
                  trait_name_to_modality_for_candidate,
                  trait_name_to_ideal_modality_for_job,
                  trait_info_df)
    
    strong_traits_data(selected_traits,
                    trait_name_to_modality_for_candidate,
                    candidate_top_values,
                    trait_info_df)
    
    get_infos(first_name, last_name, job_name)

# Generate page header
def generate_header(page: int):
    return f"""
        <div class="Header">
            <div class="HeaderIcons">
                <svg width="9" height="8" viewBox="0 0 9 8" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path
                    d="M4.48743 1.04626L7.97486 6.97489H1L4.48743 1.04626Z"
                    stroke="#D6E900"
                    stroke-width="0.97648"
                    ></path>
                </svg>
                <svg width="8" height="8" viewBox="0 0 8 8" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect
                    x="0.669098"
                    y="0.69751"
                    width="6.62611"
                    height="6.62611"
                    stroke="#2656FF"
                    stroke-width="0.97648"
                    ></rect>
                </svg>
                <svg width="9" height="10" viewBox="0 0 9 10" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path
                    d="M0.989456 5.01056C0.989456 2.9882 2.6289 1.34875 4.65126 1.34875V1.34875C6.67361 1.34875 8.31306 2.9882 8.31306 5.01056V5.01056C8.31306 7.03291 6.67361 8.67236 4.65126 8.67236V8.67236C2.6289 8.67236 0.989456 7.03291 0.989456 5.01056V5.01056Z"
                    stroke="#54E975"
                    stroke-width="0.97648"
                    ></path>
                </svg>
                <svg width="8" height="10" viewBox="0 0 8 10" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path
                    d="M7.28467 5.8824C7.28467 7.48745 5.87943 9.02109 4.14598 9.02109C2.41253 9.02109 1.00729 7.48745 1.00729 5.8824C1.00729 3.78994 4.14598 1 4.14598 1C4.14598 1 7.28467 3.78994 7.28467 5.8824Z"
                    stroke="#8B5CF6"
                    stroke-width="0.97648"
                    ></path>
                </svg>
                <svg width="9" height="8" viewBox="0 0 9 8" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path
                    d="M4.98946 0.174316L4.99089 3.75544C4.99093 3.8501 5.10438 3.89857 5.1728 3.83317L7.82534 1.29791L5.18663 3.82382C5.11658 3.89087 5.16402 4.00908 5.26098 4.00912L9 4.01049L5.26098 4.01186C5.16402 4.01189 5.11658 4.1301 5.18663 4.19715L7.82534 6.72307L5.1728 4.18781C5.10438 4.12241 4.99093 4.17088 4.99089 4.26554L4.98946 7.84666L4.98802 4.26554C4.98798 4.17088 4.87454 4.12241 4.80611 4.18781L2.15357 6.72307L4.79228 4.19715C4.86233 4.1301 4.81489 4.01189 4.71793 4.01186L0.978912 4.01049L4.71793 4.00912C4.81489 4.00908 4.86233 3.89087 4.79228 3.82382L2.15357 1.29791L4.80611 3.83317C4.87454 3.89857 4.98798 3.8501 4.98802 3.75544L4.98946 0.174316Z"
                    stroke="#FF9548"
                    stroke-width="0.97648"
                    ></path>
                </svg>
            </div>
            <div>0{page}</div>
        </div>
    """

# Generate page footer
def generate_footer(candidate_name: str):
    return f"""
        <div class="Footer">
            <div class="FooterTitle">Rapport de personnalité</div>
            <div class="FooterCandidate">{candidate_name}</div>
            <div class="FooterLogo">
            <svg width="77" height="14" viewBox="0 0 55 10" fill="none" xmlns="http://www.w3.org/2000/svg">
                <g clip-path="url(#clip0_281_6046)">
                <path
                    fill-rule="evenodd"
                    clip-rule="evenodd"
                    d="M5.89216 0.00363636C6.86326 0.00363636 7.7169 0.104545 8.45453 0.306667C9.19159 0.508485 9.80232 0.781212 10.2887 1.12424C10.774 1.46788 11.1404 1.87576 11.3887 2.34788C11.6377 2.82061 11.7611 3.32758 11.7611 3.86848V3.90485H9.72154V3.86848C9.72154 3.5697 9.62701 3.29515 9.43966 3.04455C9.25346 2.79455 8.99107 2.58242 8.65419 2.40848C8.31846 2.23515 7.90997 2.09939 7.42844 2.00273C6.94719 1.90576 6.41925 1.85727 5.84633 1.85727C4.47849 1.85727 3.50596 2.13212 2.92846 2.68152C2.35182 3.2303 2.06365 3.99394 2.06365 4.97152V5.10424C2.06365 6.05818 2.3501 6.80121 2.92273 7.33394C3.49622 7.86758 4.47133 8.13394 5.84633 8.13394C6.32013 8.13394 6.77445 8.10546 7.21044 8.04879C7.64586 7.99273 8.03602 7.8897 8.38005 7.7397C8.72352 7.59061 9.01628 7.38212 9.2569 7.11606C9.49753 6.84939 9.65995 6.51 9.74388 6.09788H5.91536V4.32879H11.7608V9.93939H10.5124L10.2366 8.09758C9.96961 8.63879 9.45456 9.09152 8.69544 9.45515C7.93404 9.81818 6.97669 10.0003 5.82427 10.0003C4.76896 10.0003 3.87063 9.88455 3.12984 9.65424C2.38792 9.42485 1.78635 9.09727 1.32372 8.67333C0.867392 8.2589 0.520051 7.72674 0.315391 7.12849C0.10513 6.52242 0 5.84333 0 5.09273V4.87424C0 4.15515 0.120313 3.49727 0.360938 2.89939C0.601563 2.30152 0.9625 1.78909 1.4449 1.36061C1.92615 0.932424 2.53917 0.599091 3.28453 0.361212C4.02904 0.122727 4.89844 0.00363636 5.89245 0.00363636H5.89216ZM17.2975 1.15455C18.9475 1.15455 20.1741 1.50424 20.9773 2.20242C21.78 2.90182 22.1808 3.85273 22.1808 5.05606V6.20727H14.477C14.5535 6.9503 14.8225 7.47152 15.2851 7.7703C15.7478 8.06939 16.4335 8.21879 17.343 8.21879C17.7114 8.22096 18.0794 8.19259 18.4436 8.13394C18.7796 8.07758 19.071 7.99 19.3153 7.87333C19.5594 7.75606 19.7499 7.60061 19.8877 7.40697C20.026 7.21273 20.0939 6.98242 20.0939 6.71606V6.69182H22.1808V6.71606C22.1808 7.17667 22.0868 7.60697 21.8992 8.00636C21.7127 8.40636 21.4216 8.75364 21.0278 9.04879C20.6342 9.34364 20.1283 9.57606 19.509 9.74576C18.8902 9.91485 18.1489 10 17.2852 10C15.6037 10 14.3662 9.62818 13.5724 8.88485C12.7766 8.14212 12.3802 7.0997 12.3802 5.75879V5.66182C12.3802 4.23242 12.7795 3.12364 13.5778 2.33636C14.3765 1.54848 15.6151 1.15455 17.2975 1.15455V1.15455ZM50.117 1.15455C51.828 1.15455 53.0707 1.54636 53.8421 2.3297C54.6139 3.11364 55 4.18333 55 5.54061V5.61333C55 6.97061 54.6139 8.04061 53.8421 8.82455C53.0707 9.60788 51.8283 9.9997 50.117 9.9997C48.4046 9.9997 47.1608 9.60788 46.3856 8.82455C45.6093 8.04091 45.2215 6.97061 45.2215 5.61333V5.54061C45.2215 4.18333 45.6096 3.11364 46.3856 2.3297C47.1611 1.54636 48.4046 1.15455 50.117 1.15455V1.15455ZM34.5738 0.0642424C35.3146 0.0642424 35.9474 0.145152 36.4716 0.306364C36.9947 0.467879 37.4212 0.694242 37.7495 0.985152C38.0708 1.26592 38.3192 1.62864 38.4714 2.03909C38.6246 2.45091 38.7008 2.90364 38.7008 3.39606V3.54152C38.7008 4.02636 38.6246 4.47061 38.4716 4.87424C38.3187 5.27848 38.0758 5.62364 37.744 5.91C37.4117 6.19727 36.9772 6.42152 36.4424 6.58273C35.9078 6.74424 35.2662 6.82515 34.5162 6.82515H31.307V9.93939H29.2437V0.0642424H34.5738ZM44.5463 1.15455C44.6918 1.15455 44.8055 1.15879 44.8897 1.16667C44.9787 1.17555 45.067 1.19176 45.1536 1.21515V3.37182C44.8021 3.29091 44.4065 3.23849 43.9676 3.21424C43.5276 3.19 43.1169 3.21424 42.7344 3.28697C42.3523 3.3597 42.0295 3.49121 41.7662 3.68091C41.5032 3.87091 41.3709 4.13939 41.3709 4.48667V9.93909H39.3072V1.21485H41.3706V4.08697C41.3706 3.71515 41.4267 3.35152 41.5367 2.99636C41.6463 2.64378 41.8307 2.32243 42.0759 2.05697C42.3242 1.78667 42.6525 1.56848 43.0613 1.40303C43.4695 1.23788 43.9653 1.15455 44.5463 1.15455V1.15455ZM26.0895 0V0.29697C26.0901 1.11667 25.897 1.94818 25.2926 2.10485H28.6166V4.0703H25.9766V7.03394C25.9766 7.85364 26.2049 7.97394 26.6378 7.97394H28.6166V9.93909H25.913C24.306 9.93909 23.998 9.25303 23.998 7.03364V4.0703H22.5222V2.10455H23.3137C23.9869 1.9003 24.1232 1.04424 24.1112 0.29697H24.1106V0H26.0898H26.0895ZM50.117 3.10545C49.1459 3.10545 48.4332 3.28727 47.9792 3.6503C47.5237 4.01394 47.2963 4.64818 47.2963 5.55273V5.60152C47.2963 6.50606 47.5237 7.14 47.9792 7.50364C48.4332 7.86727 49.1462 8.04909 50.117 8.04909C51.0876 8.04909 51.8003 7.86727 52.2543 7.50364C52.7089 7.14 52.9364 6.50606 52.9364 5.60152V5.55273C52.9364 4.64818 52.7089 4.01394 52.2546 3.6503C51.8003 3.28697 51.0876 3.10545 50.117 3.10545V3.10545ZM34.4022 2.00303H31.307V5.00758H34.4022C35.1355 5.00758 35.6787 4.89455 36.0293 4.66818C36.3816 4.44242 36.5572 4.05848 36.5572 3.51758V3.48091C36.5572 2.94758 36.3819 2.56848 36.0293 2.34212C35.6787 2.11576 35.1355 2.00273 34.4022 2.00273V2.00303ZM17.2517 2.88727C16.4791 2.88727 15.8718 3.00848 15.4286 3.25091C14.9846 3.49333 14.6953 3.90121 14.5575 4.47455H20.083C19.9905 3.91697 19.7144 3.51333 19.2517 3.26303C18.7888 3.01242 18.1231 2.88727 17.2517 2.88727V2.88727Z"
                    fill="white"
                ></path>
                </g>
                <defs>
                <clipPath id="clip0_281_6046"><rect width="55" height="10" fill="white"></rect></clipPath>
                </defs>
            </svg>
            </div>
        </div>
    """

# Generate cover page
def generate_cover(candidate_name: str, date: str):
    return f"""
        <section class="CouvertureSection">
            <div class="CouvertureLogo">
                <svg width="100" height="17.5" viewBox="0 0 55 10" fill="none" xmlns="http://www.w3.org/2000/svg">
                <g clip-path="url(#clip0_281_6046)">
                    <path
                    fill-rule="evenodd"
                    clip-rule="evenodd"
                    d="M5.89216 0.00363636C6.86326 0.00363636 7.7169 0.104545 8.45453 0.306667C9.19159 0.508485 9.80232 0.781212 10.2887 1.12424C10.774 1.46788 11.1404 1.87576 11.3887 2.34788C11.6377 2.82061 11.7611 3.32758 11.7611 3.86848V3.90485H9.72154V3.86848C9.72154 3.5697 9.62701 3.29515 9.43966 3.04455C9.25346 2.79455 8.99107 2.58242 8.65419 2.40848C8.31846 2.23515 7.90997 2.09939 7.42844 2.00273C6.94719 1.90576 6.41925 1.85727 5.84633 1.85727C4.47849 1.85727 3.50596 2.13212 2.92846 2.68152C2.35182 3.2303 2.06365 3.99394 2.06365 4.97152V5.10424C2.06365 6.05818 2.3501 6.80121 2.92273 7.33394C3.49622 7.86758 4.47133 8.13394 5.84633 8.13394C6.32013 8.13394 6.77445 8.10546 7.21044 8.04879C7.64586 7.99273 8.03602 7.8897 8.38005 7.7397C8.72352 7.59061 9.01628 7.38212 9.2569 7.11606C9.49753 6.84939 9.65995 6.51 9.74388 6.09788H5.91536V4.32879H11.7608V9.93939H10.5124L10.2366 8.09758C9.96961 8.63879 9.45456 9.09152 8.69544 9.45515C7.93404 9.81818 6.97669 10.0003 5.82427 10.0003C4.76896 10.0003 3.87063 9.88455 3.12984 9.65424C2.38792 9.42485 1.78635 9.09727 1.32372 8.67333C0.867392 8.2589 0.520051 7.72674 0.315391 7.12849C0.10513 6.52242 0 5.84333 0 5.09273V4.87424C0 4.15515 0.120313 3.49727 0.360938 2.89939C0.601563 2.30152 0.9625 1.78909 1.4449 1.36061C1.92615 0.932424 2.53917 0.599091 3.28453 0.361212C4.02904 0.122727 4.89844 0.00363636 5.89245 0.00363636H5.89216ZM17.2975 1.15455C18.9475 1.15455 20.1741 1.50424 20.9773 2.20242C21.78 2.90182 22.1808 3.85273 22.1808 5.05606V6.20727H14.477C14.5535 6.9503 14.8225 7.47152 15.2851 7.7703C15.7478 8.06939 16.4335 8.21879 17.343 8.21879C17.7114 8.22096 18.0794 8.19259 18.4436 8.13394C18.7796 8.07758 19.071 7.99 19.3153 7.87333C19.5594 7.75606 19.7499 7.60061 19.8877 7.40697C20.026 7.21273 20.0939 6.98242 20.0939 6.71606V6.69182H22.1808V6.71606C22.1808 7.17667 22.0868 7.60697 21.8992 8.00636C21.7127 8.40636 21.4216 8.75364 21.0278 9.04879C20.6342 9.34364 20.1283 9.57606 19.509 9.74576C18.8902 9.91485 18.1489 10 17.2852 10C15.6037 10 14.3662 9.62818 13.5724 8.88485C12.7766 8.14212 12.3802 7.0997 12.3802 5.75879V5.66182C12.3802 4.23242 12.7795 3.12364 13.5778 2.33636C14.3765 1.54848 15.6151 1.15455 17.2975 1.15455V1.15455ZM50.117 1.15455C51.828 1.15455 53.0707 1.54636 53.8421 2.3297C54.6139 3.11364 55 4.18333 55 5.54061V5.61333C55 6.97061 54.6139 8.04061 53.8421 8.82455C53.0707 9.60788 51.8283 9.9997 50.117 9.9997C48.4046 9.9997 47.1608 9.60788 46.3856 8.82455C45.6093 8.04091 45.2215 6.97061 45.2215 5.61333V5.54061C45.2215 4.18333 45.6096 3.11364 46.3856 2.3297C47.1611 1.54636 48.4046 1.15455 50.117 1.15455V1.15455ZM34.5738 0.0642424C35.3146 0.0642424 35.9474 0.145152 36.4716 0.306364C36.9947 0.467879 37.4212 0.694242 37.7495 0.985152C38.0708 1.26592 38.3192 1.62864 38.4714 2.03909C38.6246 2.45091 38.7008 2.90364 38.7008 3.39606V3.54152C38.7008 4.02636 38.6246 4.47061 38.4716 4.87424C38.3187 5.27848 38.0758 5.62364 37.744 5.91C37.4117 6.19727 36.9772 6.42152 36.4424 6.58273C35.9078 6.74424 35.2662 6.82515 34.5162 6.82515H31.307V9.93939H29.2437V0.0642424H34.5738ZM44.5463 1.15455C44.6918 1.15455 44.8055 1.15879 44.8897 1.16667C44.9787 1.17555 45.067 1.19176 45.1536 1.21515V3.37182C44.8021 3.29091 44.4065 3.23849 43.9676 3.21424C43.5276 3.19 43.1169 3.21424 42.7344 3.28697C42.3523 3.3597 42.0295 3.49121 41.7662 3.68091C41.5032 3.87091 41.3709 4.13939 41.3709 4.48667V9.93909H39.3072V1.21485H41.3706V4.08697C41.3706 3.71515 41.4267 3.35152 41.5367 2.99636C41.6463 2.64378 41.8307 2.32243 42.0759 2.05697C42.3242 1.78667 42.6525 1.56848 43.0613 1.40303C43.4695 1.23788 43.9653 1.15455 44.5463 1.15455V1.15455ZM26.0895 0V0.29697C26.0901 1.11667 25.897 1.94818 25.2926 2.10485H28.6166V4.0703H25.9766V7.03394C25.9766 7.85364 26.2049 7.97394 26.6378 7.97394H28.6166V9.93909H25.913C24.306 9.93909 23.998 9.25303 23.998 7.03364V4.0703H22.5222V2.10455H23.3137C23.9869 1.9003 24.1232 1.04424 24.1112 0.29697H24.1106V0H26.0898H26.0895ZM50.117 3.10545C49.1459 3.10545 48.4332 3.28727 47.9792 3.6503C47.5237 4.01394 47.2963 4.64818 47.2963 5.55273V5.60152C47.2963 6.50606 47.5237 7.14 47.9792 7.50364C48.4332 7.86727 49.1462 8.04909 50.117 8.04909C51.0876 8.04909 51.8003 7.86727 52.2543 7.50364C52.7089 7.14 52.9364 6.50606 52.9364 5.60152V5.55273C52.9364 4.64818 52.7089 4.01394 52.2546 3.6503C51.8003 3.28697 51.0876 3.10545 50.117 3.10545V3.10545ZM34.4022 2.00303H31.307V5.00758H34.4022C35.1355 5.00758 35.6787 4.89455 36.0293 4.66818C36.3816 4.44242 36.5572 4.05848 36.5572 3.51758V3.48091C36.5572 2.94758 36.3819 2.56848 36.0293 2.34212C35.6787 2.11576 35.1355 2.00273 34.4022 2.00273V2.00303ZM17.2517 2.88727C16.4791 2.88727 15.8718 3.00848 15.4286 3.25091C14.9846 3.49333 14.6953 3.90121 14.5575 4.47455H20.083C19.9905 3.91697 19.7144 3.51333 19.2517 3.26303C18.7888 3.01242 18.1231 2.88727 17.2517 2.88727V2.88727Z"
                    fill="white"
                    ></path>
                </g>
                <defs>
                    <clipPath id="clip0_281_6046"><rect width="55" height="10" fill="white"></rect></clipPath>
                </defs>
                </svg>
            </div>
            <div class="CouvertureTitle">
                <div>
                    <div class="Candidate">{candidate_name.replace(" ", "<br/>")}</div>
                    <div class="Icons">
                        <svg width="24" height="22" viewBox="0 0 9 8" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path
                            d="M4.48743 1.04626L7.97486 6.97489H1L4.48743 1.04626Z"
                            stroke="#D6E900"
                            stroke-width="0.97648"
                            ></path>
                        </svg>
                        <svg width="22" height="22" viewBox="0 0 8 8" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <rect
                            x="0.669098"
                            y="0.69751"
                            width="6.62611"
                            height="6.62611"
                            stroke="#2656FF"
                            stroke-width="0.97648"
                            ></rect>
                        </svg>
                        <svg width="24" height="24" viewBox="0 0 9 10" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path
                            d="M0.989456 5.01056C0.989456 2.9882 2.6289 1.34875 4.65126 1.34875V1.34875C6.67361 1.34875 8.31306 2.9882 8.31306 5.01056V5.01056C8.31306 7.03291 6.67361 8.67236 4.65126 8.67236V8.67236C2.6289 8.67236 0.989456 7.03291 0.989456 5.01056V5.01056Z"
                            stroke="#54E975"
                            stroke-width="0.97648"
                            ></path>
                        </svg>
                        <svg width="22" height="24" viewBox="0 0 8 10" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path
                            d="M7.28467 5.8824C7.28467 7.48745 5.87943 9.02109 4.14598 9.02109C2.41253 9.02109 1.00729 7.48745 1.00729 5.8824C1.00729 3.78994 4.14598 1 4.14598 1C4.14598 1 7.28467 3.78994 7.28467 5.8824Z"
                            stroke="#8B5CF6"
                            stroke-width="0.97648"
                            ></path>
                        </svg>
                        <svg width="24" height="24" viewBox="0 0 9 8" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path
                            d="M4.98946 0.174316L4.99089 3.75544C4.99093 3.8501 5.10438 3.89857 5.1728 3.83317L7.82534 1.29791L5.18663 3.82382C5.11658 3.89087 5.16402 4.00908 5.26098 4.00912L9 4.01049L5.26098 4.01186C5.16402 4.01189 5.11658 4.1301 5.18663 4.19715L7.82534 6.72307L5.1728 4.18781C5.10438 4.12241 4.99093 4.17088 4.99089 4.26554L4.98946 7.84666L4.98802 4.26554C4.98798 4.17088 4.87454 4.12241 4.80611 4.18781L2.15357 6.72307L4.79228 4.19715C4.86233 4.1301 4.81489 4.01189 4.71793 4.01186L0.978912 4.01049L4.71793 4.00912C4.81489 4.00908 4.86233 3.89087 4.79228 3.82382L2.15357 1.29791L4.80611 3.83317C4.87454 3.89857 4.98798 3.8501 4.98802 3.75544L4.98946 0.174316Z"
                            stroke="#FF9548"
                            stroke-width="0.97648"
                            ></path>
                        </svg>
                    </div>
                </div>
            </div>
            <div class="CouvertureFooter">
                <div class="CouvertureFooterTitle">Rapport d’adéquation<br />à un poste</div>
                <div class="CouvertureFooterDate">{date}</div>
            </div>
        </section>
    """


# Generate about page
def generate_about(candidate_name: str, job_name: str):
    return f"""
        <section>
            {generate_header(1)}
            <div class="AboutTitleWrapper">
                <div class="AboutTitle">À PROPOS</div>
                <div class="AboutSubTitle">DU RAPPORT D’ADÉQUATION À UN POSTE</div>
            </div>
            <div class="AboutContent">
                Ce rapport est issu de la fiche de poste <b style="text-transform: capitalize">{job_name}</b> et des réponses de
                <b style="text-transform: capitalize">{candidate_name}</b> au Questionnaire Big Five GetPro.<br /><br />Il vise
                à présenter l'adéquation entre les <b>Softkills recherchées pour le poste</b> et les
                <b>traits de personnalité du candidat</b>. Il est important de noter qu'un candidat peut très bien performer à
                un poste sans correspondre exactement au profil type ; cependant, il est important d’<b
                >analyser ces divergences</b
                >.<br /><br />La liste des traits de personnalité mentionnés sont ceux attendus pour le poste et l’indicateur
                (“élevé”, “très élevé”, “bas”, “très bas”) correspond au score du candidat. Par exemple, "Leadership élevé"
                indique que le candidat possède un fort leadership. Si cette caractéristique est marquée en vert, cela signifie
                qu'un leadership élevé est souhaité pour le poste. En revanche, si elle est en orange, le poste requiert un
                niveau de leadership bas.<br /><br />Pour chaque trait de personnalité mentionné, nous proposons des
                <b>questions d’entretien</b> afin d'approfondir la discussion. Ces questions sont particulièrement utiles pour
                <b>explorer les écarts entre les traits de personnalité du candidat et les exigences du poste</b>, indiqués en
                orange.<br /><br />Pour toute remarque, n’hésitez pas à envoyer un mail à
                <u>science@getpro.fr.</u>
            </div>
            {generate_footer(candidate_name)}
        </section>
    """

def generate_block_jobmatch(trait_modality: str, fit: bool, question: str):
    fit_class = "BlockValid" if fit else "BlockInvalid"
    icon_content = f"""<svg width="12" height="12" viewBox="0 0 8 8" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path
                  d="M1.33331 3.83333L3.16449 5.66667L6.83331 2"
                  stroke="#93C5FD"
                  stroke-width="1.2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                ></path>
              </svg>""" if fit else ""
    return f"""
        <div class="Block {fit_class}">
          <div class="Trait">
            <div class="TraitIcon">
              {icon_content}
            </div>
            <div class="TraitModality">{trait_modality}</div>
          </div>
          <p class="Question">
            ➤ {question}
          </p>
        </div>
    """

def generate_block_trait(trait_modality: str, question: str):
    return f"""
        <div class="Block BlockTrait">
          <div class="Trait">
            <div>{trait_modality}</div>
          </div>
          <p class="Question">
            ➤ {question}
          </p>
        </div>
    """

def generate_list_page(candidate_name: str, page: int, title: str, elements: list, kind: str):
    title_element = f"<h1 class='Title'>{title}</h1>" if len(title) > 0 else "" 
    section = f"<section>{generate_header(page)}{title_element}"
    section += "<div class='Questions'>"
    for question in elements:
        if kind == "jobmatchs":
            section += f"{generate_block_jobmatch(question['trait_modality'], question['fit'], question['question'])}"
        elif kind == "traits":
            section += f"{generate_block_trait(question['trait_modality'], question['question'])}"
    section += "</div>"
    section += f"{generate_footer(candidate_name)}</section>"
    return section

def list_cutter(ls):
    mini_list = []
    new_list = []

    for index, element in enumerate(ls):
        if index == 11 or (index > 11 and (index - 11) % 13 == 0):
            new_list.append(mini_list)
            mini_list = []
            mini_list.append(element)
        
        else:
            mini_list.append(element)

    if len(mini_list) > 0:
        new_list.append(mini_list)
    
    return new_list

def generate_document(candidate_first_name: str, candidate_last_name: str, job_name: str, job_matchs: list, traits: list):
    file_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="" />
        <link
        href="https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400;1,500;1,600;1,700&amp;display=swap"
        rel="stylesheet"
        />
        <style>
        /* Couverture */

        .CouvertureSection {{
            display: flex;
            flex-direction: column;
        }}

        .CouvertureLogo {{
            padding: 80px;
        }}

        .CouvertureFooter {{
            display: flex;
            align-items: end;
            justify-content: space-between;
            padding: 80px;
        }}

        .CouvertureFooterTitle {{
            font-size: 32px;
            line-height: 36px;
            font-weight: 700;
        }}

        .CouvertureFooterDate {{
            font-weight: 700;
        }}

        .CouvertureTitle {{
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        .Candidate {{
            font-size: 52px;
            line-height: 62px;
            font-weight: 500;
            text-align: center;
            text-transform: capitalize;
        }}

        .Icons {{
            margin-top: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .Icons > svg {{
            margin-left: 16px;
            margin-right: 16px;
        }}

        /* About */

        .AboutTitleWrapper {{
            padding-top: 40px;
            padding-bottom: 40px;
        }}

        .AboutTitleWrapper > .AboutTitle {{
            text-align: center;
            font-weight: 700;
            font-size: 32px;
            line-height: 36px;
            margin-bottom: 10px;
        }}

        .AboutTitleWrapper > .AboutSubTitle {{
            text-align: center;
            font-size: 18px;
            line-height: 24px;
            font-weight: 400;
            color: #cbd5e1;
        }}

        .AboutContent {{
            padding-left: 60px;
            padding-right: 60px;
            color: #cbd5e1;
            line-height: 24px;
            font-size: 13px;
        }}

        .AboutContent > b {{
            color: white;
        }}
        
        /* Jobmatch */

        .Questions {{
            color: white;
            padding: 15px 20px;
            overflow: hidden;
        }}

        .Title {{
            color: white;
            padding-top: 15px;
            height: 90px;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            font-size: 32px;
            padding-left: 100px;
            padding-right: 100px;
        }}

        .Block {{
            height: 50px;
            border-radius: 15px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            padding: 10px 20px;
        }}

        .BlockInvalid {{
            background: rgba(38, 85, 255, 0.2);
        }}

        .BlockValid,
        .BlockTrait {{
            background: #0f1f56;
        }}

        .Trait {{
            font-size: 13px;
            font-weight: 700;
            width: 30%;
            display: flex;
            align-items: center;
        }}

        .TraitIcon {{
            width: 14px;
            height: 14px;
            min-width: 14px;
            min-height: 14px;
            border-radius: 4px;
            position: relative;
            margin-right: 12px;
            border: 1px rgba(255, 255, 255, 0.32) solid;
        }}

        .TraitModality {{
            flex: 1;
        }}

        .TraitIcon > svg {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%)
        }}

        .Question {{
            font-size: 10px;
            font-weight: 400;
            flex: 1;
            color: #f1f5f9;
            margin-left: 20px;
        }}

        .Header {{
            padding-bottom: 10px;
            margin-left: 20px;
            margin-right: 20px;
            margin-top: 15px;
            display: flex;
            justify-content: space-between;
            border-bottom: 2px #2b365c solid;
            color: #f1f5f9;
            font-size: 12px;
            font-weight: 700;
        }}

        .HeaderIcons {{
        }}

        .HeaderIcons > svg {{
            margin-left: 6px;
            margin-right: 6px;
        }}

        .Footer {{
            position: absolute;
            bottom: 0px;
            left: 0px;
            display: flex;
            margin-left: 20px;
            margin-right: 20px;
            margin-bottom: 15px;
            padding-top: 10px;
            border-top: 2px #2b365c solid;
            width: 95%;
            color: white;
            align-items: center;
        }}

        .FooterTitle {{
            flex: 1;
            font-weight: 500;
            font-size: 10px;
        }}

        .FooterCandidate {{
            flex: 1;
            text-align: center;
            text-transform: uppercase;
            font-weight: 700;
            font-size: 12px;
        }}

        .FooterLogo {{
            flex: 1;
            display: flex;
            justify-content: end;
        }}

        .Label {{
            padding: 8px 16px;
            background: white;
            border-radius: 4px;
            cursor: pointer;
        }}

        .LabelWrapper {{
            margin-bottom: 20px;
        }}

        body {{
            height: 100%;
            width: 100%;
            margin: 0px;
            background: #0f133f;
        }}
        section {{
            height: 297mm;
            width: 210mm;
            clear: both;
            page-break-after: always;
            position: relative;
            overflow: hidden;
            font-family: "Montserrat";
            color: white;
        }}
        @page {{
            margin: 0mm;
            height: 297mm;
            width: 210mm;
        }}
        </style>
    </head>
    <body>
    </body>
    </html>"""

    today = date.today().strftime("%d - %m - %Y")
    candidate_name = candidate_first_name + " " + candidate_last_name

    file_content += generate_cover(candidate_name, today)
    file_content += generate_about(candidate_name, job_name)

    job_match_lists = list_cutter(job_matchs)
    for index, elements in enumerate(job_match_lists):
        title = "Points à valider" if index == 0 else "" 
        file_content += generate_list_page(candidate_name, index + 2, title, elements, "jobmatchs")

    traits_list = list_cutter(traits)
    for index, elements in enumerate(traits_list):
        title = f"Partie spécifique aux traits marqués de {candidate_first_name}" if index == 0 else ""
        file_content += generate_list_page(candidate_name, index + 2 + len(job_match_lists), title, elements, "traits")

    file_content += "</body></html>"
    html_file_path = f"./{candidate_first_name}_{candidate_last_name}_report.html"
    with open(html_file_path, 'w', encoding='utf-8') as file:
        file.write(file_content)
    HTML(html_file_path).write_pdf(html_file_path.replace("html", "pdf"))



# Generate file
def generate_file():
    with open('output_data/jobmatch_data.json', 'r') as file:
        job_matchs = json.load(file)

    with open('output_data/traits_data.json', 'r') as file:
        traits = json.load(file)

    with open('output_data/infos.json', 'r') as file:
        infos = json.load(file)

    candidate_first_name = infos['first_name']
    candidate_last_name = infos['last_name']
    job_name = infos['job_name']

    generate_document(candidate_first_name, candidate_last_name, job_name, job_matchs, traits)