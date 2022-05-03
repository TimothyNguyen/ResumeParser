from cgitb import text
import io
import os
import re
import nltk
import spacy
import pandas as pd
import constants as cs
import spacy
from spacy.matcher import PhraseMatcher
from flask import Flask, request, jsonify
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# load default skills data base
from skillNer.general_params import SKILL_DB
# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor

nlp = spacy.load("en_core_web_md")
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

def job_description_preprocess(raw_text):
    full_string = re.sub(r'\n+','\n',raw_text)
    full_string = full_string.replace("\r", "\n")
    full_string = full_string.replace("\t", "\n")
    full_string = full_string.replace("|", "\n")
    full_string = re.sub(r'[ \t]{3,}', '\n', full_string)

    return full_string

def extract_skills(jd_text):
    # init skill extractor

    # extract skills from job_description
    annotations = skill_extractor.annotate(jd_text)
    skills = annotations['results']['full_matches'] + annotations['results']['ngram_scored']
    skills = set(elem['doc_node_value'].lower() for elem in skills) 
    return list(skills)

def extract_degree(resume_text):
    '''
    Helper function to extract education from spacy nlp text
    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :return: tuple of education degree and year if year if found
             else only returns education degree
    '''
    nlp = spacy.load("en_core_web_sm")
    nlp_text = nlp(resume_text)
    # print(nlp_text)
    # Sentence Tokenizer
    nlp_text = [sent.text.strip() for sent in nlp_text.sents]
    # print(nlp_text)
    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        #print(index, text), print('-'*50)
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            if tex.upper() in cs.EDUCATION and tex not in cs.STOPWORDS:
                return [resume_text]
                # edu[tex] = text + nlp_text[index]
    return list()
# print(extract_degree("\n".join(entities['education'])))
RESERVED_WORDS = [
    'school',
    'college',
    'univers',
    'academy',
    'faculty',
    'institute',
    'faculdades',
    'Schola',
    'schule',
    'lise',
    'lyceum',
    'lycee',
    'polytechnic',
    'kolej',
    'ünivers',
    'okul',
    'tech'
]

def extract_education_section(text):
    nlp = spacy.load('en_core_web_sm')

    def get_gpa(text):
        rx = re.compile(r'(\bgpa[ :]+)?(\d+(?:\.\d+)?)[/\d. ]{0,6}(?(1)| *gpa\b)', re.I)
        m = rx.search(text)
        if m:
            return m.group(2)
        return ""

    degree_list = set(extract_degree(text))
    # extracting entities
    new_gpa = get_gpa(text)
    return degree_list, new_gpa

data = {
    "profile": {
        "education": [
            {
                "degree": "Bachelor of Science in Computer Science, Statistics & Data Science",
                "description": "GPA: 3.835\nRelevant Coursework: Algorithms, Data Structures, Software Engineering, Scalable Web Systems, Computer Network Theory,\nDatabases, Information Retrieval, Data Science, Natural Language Processing, Artificial Intelligence\nAmherst, MA\n",
                "from": "",
                "gpa": "3.835",
                "to": "05/2022",
                "university": "University of Massachusetts, Amherst"
            }
        ],
        "email": "quynhthoa1972@gmail.com",
        "experience": [
            {
                "company": "MathWorks",
                "description": "● Modernizing & Developing JavaScript Web Components for MathWorks products with HTML, CSS, JS & React.\n",
                "from": "01/2021",
                "title": "Software Engineering Intern",
                "to": "12/2021"
            },
            {
                "company": "Dell Technologies",
                "description": "● Collaborated to prototype edge-computing platforms with Litmus Edge and VMware to optimize the manufacturing industry.\n● Capture, analyze, visualize, and manage industrial data with Python, Tensorflow, JS, Docker, Postgres, TimescaleDB, and Grafana.\n● Innovated ML/Time Series models such as Kalman Filters and SARIMAX to estimate noisy trends from industrial MQTT sensors.\n● Monitored and performance tested Litmus Edge within VMware, supporting up to 10,000 industrial automation devices.\n",
                "from": "05/2021",
                "title": "Software Engineering Intern",
                "to": "08/2021"
            },
            {
                "company": "Systems & Technology Research",
                "description": "● Designed a Spring Boot/Java REST service to rapidly automate assurance cases & causal models for predictive maintenance.\n● Developed Bayesian Networks to support causal model architecture and Uber Model Manager to manage different models.\nBUILD UMass\n",
                "from": "05/2020",
                "title": "Software Engineering Intern",
                "to": "08/2020"
            },
            {
                "company": "Systems & Technology Research",
                "description": "Github: https://git.io/JLpVa\n● Provide non-profits, startups, and local businesses with web and mobile applications through pro-bono engagements.\n● Grew organization to over 50 software developers, 20 business developers, and partnered with over 10 clients over two years.\n● Led two teams of 8 people with the cofounder of BUILD UMass to develop an auth/user management & forms/surveys system\n● Developing JSON form-building application to customize form-making, filling, as well as drag-and-drop capabilities.\n● Employed React for front-end UI, Node/Express.js/MongoDB for backend, along with Redux & Passport/JWT Integration.\nDSC-WAV (National Science Foundation)\n",
                "from": "01/2019",
                "title": "Software Engineer",
                "to": "Present"
            },
            {
                "company": "Systems & Technology Research",
                "description": "Github: https://git.io/JLpah\n● Research to provide better transportation for people with drug addiction to get to treatment centers on an eight-person team.\n● Developed visualizations and analyzed the cost-time benefit to get to treatment centers with Python and Google Cloud.\n",
                "from": "01/2020",
                "title": "Data Scientist",
                "to": "01/2021"
            }
        ],
        "name": "Timothy Nguyen",
        "phone": "6178883076",
        "skills": [
            "Mobile",
            "Software engineering",
            "Analyze",
            "Html",
            "Json",
            "Prototype",
            "Vmware",
            "Sql",
            "Automation",
            "C",
            "Docker",
            "Css",
            "R",
            "Analytics",
            "Rest",
            "Transportation",
            "Tensorflow",
            "Java",
            "Numpy",
            "Jira",
            "Computer science",
            "Linux",
            "Statistics",
            "System",
            "Architecture",
            "Programming",
            "User stories",
            "Github",
            "Engineering",
            "Ui",
            "Javascript",
            "Js",
            "Python",
            "Algorithms",
            "Cloud",
            "Spark",
            "Research",
            "Design",
            "Pandas"
        ]
    },
    "jd": """
    Who We Are

Shape a brighter financial future with us.

Together with our members, we’re changing the way people think about and interact with personal finance.

We’re a next-generation fintech company using innovative, mobile-first technology to help our nearly 3 million members reach their goals. The industry is going through an unprecedented transformation, and we’re at the forefront.

We’re proud to come to work every day knowing that what we do has a direct impact on people’s lives, with our core values guiding us every step of the way. Join us to invest in yourself, your career, and the financial world.

We are seeking a Data Engineer to join our Data Warehouse team and work in a fast-paced environment. Ideal candidates must be enthused about writing SQL and Python, solving complex data problems, good data modeling, as well as building automated data pipelines. They must be a quick learner, self-starting, and have the ability to maintain and build within a horizontal and vertical scaling data warehouse. The role will require hands-on solid experience with handling data, understanding and experience with troubleshooting and fixing data issues, as well as provisioning data pipelines and reports.

You’ll need to demonstrate first-rate attention to detail, and the ability to work quickly and accurately under pressure. You’ll need to be comfortable in developing ideas to solve problems, then executing on those ideas though deployment. In addition to the needed technical skills, the engineer will need strong verbal and written communication skills to interface with many areas of the business.

Responsibilities
Design and develop scalable data models where complex data relationships exist
Build and maintain highly automated and scalable ETL/ELT data pipelines
Communicate well with project leads, product managers, software developers, and data consumers
Provision, optimize and maintain data feeds to external systems
Create visualization and reporting as needed
Write code to validate data quality and clean existing data
Troubleshoot and solve data inconsistency issues
Be part of an on call support rotation to support the Data Warehouse and it’s automated processes
Creating well written technical documentation
Help analytics team, upstream engineering teams, as well as non-technical business users in understanding the Data Warehouse
What You’ll Need
1-2 years working experience working with automated scripting, data modeling, and data architecture (in projects or internships)
Proficient in writing and optimizing SQL scripts
Understands database architecture
Working experience in the Python language with an emphasis on data
Working knowledge of some AWS data technologies
Understanding of the software development lifecycle pr.\ocess
Skills and experience in finding, investigating, and resolving data quality issues
Ability to work in a fast-paced environment, meet deadlines, and prioritize a workload
Ability to bring new ideas and promote process improvement
Strong business communication skills that can break down technical problems into business language for non-technical personnel
BS or MS in Computer Science or related technical field
Have no more than 12 months of professional full-time work experience.
Reside in the United States and/or attend a university in the US.
Must be able to start by June 2022.
Nice To Have
Experience using business intelligence reporting tools (Tableau, Looker, etc.)
Experience writing SQL against several different database platforms
Experience creating data pipelines using Python scripting
Experience using cloud data technologies such as Redshift, Snowflake, or GCP
Experience using AWS data technologies such as (S3, Glue, Kinesis, Lambda, etc.)
Experience in building data feeds and business reports
Experience in docker
Experience using kafka
Interest in personal finance
Why You'll Love Working Here

> Internship Benefits: View Link

> Full Time University Grad Benefits: View Link

Internship Highlights Include

In addition to a unique and challenging professional experience, interns will also receive perks such as invites to Member events, and more!
Personalized Mentorship Program
Fully stocked kitchen (snacks & drinks)
Access to senior leadership, including round tables with executives across all functions of SoFi
SoFi provides equal employment opportunities (EEO) to all employees and applicants for employment without regard to race, color, religion (including religious dress and grooming practices), sex (including pregnancy, childbirth and related medical conditions, breastfeeding, and conditions related to breastfeeding), gender, gender identity, gender expression, national origin, ancestry, age (40 or over), physical or medical disability, medical condition, marital status, registered domestic partner status, sexual orientation, genetic information, military and/or veteran status, or any other basis prohibited by applicable state or federal law.

The health and safety of our employees and their families is our top priority. Due to the ongoing nature of the COVID-19 pandemic, and because unvaccinated employees pose a direct threat to the health and safety of others in the workplace, effective on November 1, 2021, U.S. employees must be fully vaccinated to work from any of our offices, travel for business or attend work-related meetings.

The company will make reasonable accommodations when possible for employees who are unable to be vaccinated because of a disability, pregnancy, sincerely held religious belief, or for other legally required reasons.

Pursuant to the San Francisco Fair Chance Ordinance, we will consider for employment qualified applicants with arrest and conviction records.

New York applicants: Notice of Employee Rights

SoFi is committed to embracing diversity. As part of this commitment, SoFi offers reasonable accommodations to candidates with physical or mental disabilities. If you need accommodations to participate in the job application or interview process, please let your recruiter know or email accommodations@sofi.com.

Due to insurance coverage issues, we are unable to accommodate remote work from Hawaii or Alaska at this time.
    """
}
textcv, jd = data['profile']['skills'], data['jd']
# print(textcv)
# print(jd)


# print(resume)
# Make call to backend of profile api
# Get info and score based on resume - use the video
jd_text = job_description_preprocess(jd)
skills = extract_skills(jd_text)

textcv = [each_string.lower() for each_string in textcv]
# print(skills)
# print(data['profile']['skills'])

# x =  '{"name":"John","age":30,"skills":["Java", "Python", "CSS"] }'
# y =  '{"Job Title":" Google Software Engineer","Description": "Learn to Code" ,"skills":["Java", "Python", "CSS"] }'
totalScore = 0
# textcv = json.loads(resume)
# textjd = json.loads(jd)
print(textcv)
print(skills)
documents = [textcv, skills]


similar_skills = set.intersection(set(textcv), set(skills))
different_skills = set.difference(set(skills), set(textcv))
count_vectorizer = CountVectorizer(analyzer=lambda x: x)
sparse_matrix = count_vectorizer.fit_transform(documents)

doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix, 
            columns=count_vectorizer.get_feature_names(), 
            index=['textjd', 'textcv'])
# df.head()
answer = cosine_similarity(df, df)
answer = pd.DataFrame(answer)
answer = answer.iloc[[1],[0]].values[0]
answer = round(float(answer),4)*100
print(answer)
print(similar_skills)
print(different_skills)