# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew

import os
from utils import get_openai_api_key

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'


vacancy_collector = Agent(
    role="Vacancy Collector",
    goal="Collect information about available data-related vacancies in the Netherlands",
    backstory="You're working on collecting information about available vacancies "
              "related to data vacancy in the Netherlands. "
              "You gather information that helps potential candidates "
              "understand the job requirements and company culture. "
              "Your work is the basis for "
              "the Candidate Finder to find a suitable candidate for these vacancies.",
    allow_delegation=False,
    verbose=True
)

candidate_finder = Agent(
    role="Candidate Finder",
    goal="Find a candidate with the correct profile for the vacancies",
    backstory="You're working on finding a suitable candidate "
              "for the vacancies collected by the Vacancy Collector. "
              "You base your search on the job requirements and company culture "
              "provided by the Vacancy Collector. "
              "You follow the main objectives and "
              "direction of the job description, "
              "as provided by the Vacancy Collector. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provided by the Vacancy Collector. "
              "You acknowledge in your search "
              "when your statements are opinions "
              "as opposed to objective statements.",
    allow_delegation=False,
    verbose=True
)

match_checker = Agent(
    role="Match Checker",
    goal="Check if candidates match with the profile",
    backstory="You're working on checking if the candidates found "
              "by the Candidate Finder match the profile for the vacancies. "
              "You base your checks on the candidate profiles and job requirements "
              "provided by the Candidate Finder and Vacancy Collector respectively. "
              "You follow the main objectives and "
              "direction of the job description and candidate profile, "
              "as provided by the Vacancy Collector and Candidate Finder. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provided by the Vacancy Collector and Candidate Finder. "
              "You acknowledge in your checks "
              "when your statements are opinions "
              "as opposed to objective statements.",
    allow_delegation=False,
    verbose=True
)


# Task for the Vacancy Collector
collect_vacancies = Task(
    description=(
        "1. Search for the latest data-related vacancies in the Netherlands.\n"
        "2. Gather detailed information about the job requirements and company culture.\n"
        "3. Develop a comprehensive list of vacancies with all the necessary details.\n"
        "4. Include any additional information that might be useful for potential candidates."
    ),
    expected_output="A comprehensive list of data-related vacancies in the Netherlands "
        "with detailed job descriptions and company information.",
    agent=vacancy_collector,
)

# Task for the Candidate Finder
find_candidates = Task(
    description=(
        "1. Search for candidates with the required profile for the vacancies.\n"
        "2. Gather detailed information about the candidates' skills and experience.\n"
        "3. Develop a comprehensive list of suitable candidates with all the necessary details.\n"
        "4. Include any additional information that might be useful for the hiring process."
    ),
    expected_output="A comprehensive list of suitable candidates "
        "with detailed profiles and contact information.",
    agent=candidate_finder,
)

# Task for the Match Checker
check_matches = Task(
    description=(
        "1. Compare the profiles of the candidates with the job requirements of the vacancies.\n"
        "2. Identify the candidates that match the profile for the vacancies.\n"
        "3. Develop a comprehensive list of matches with all the necessary details.\n"
        "4. Include any additional information that might be useful for the hiring process."
    ),
    expected_output="A comprehensive list of matches "
        "with detailed information about the candidates and vacancies.",
    agent=match_checker,
)


crew = Crew(
    agents=[vacancy_collector, candidate_finder, match_checker],
    tasks=[collect_vacancies, find_candidates, check_matches],
    verbose=2
)