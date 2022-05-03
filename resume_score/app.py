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

# init params of skill extractor
nlp = spacy.load("en_core_web_md")
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
app = Flask(__name__)

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
    skills = set(elem['doc_node_value']for elem in skills) 
    return list(skills)

@app.route('/resumescore', methods=['GET'])
def hello():
    data = request.get_json()
    textcv, jd = data['profile'], data['jd']
    skills_resume = textcv['skills'] 
    # print(resume)
    # Make call to backend of profile api
    # Get info and score based on resume - use the video
    jd_text = job_description_preprocess(jd)
    skills = extract_skills(jd_text)

    skills_resume = [each_string.lower() for each_string in skills_resume]
    # print(skills)
    # print(data['profile']['skills'])

    # x =  '{"name":"John","age":30,"skills":["Java", "Python", "CSS"] }'
    # y =  '{"Job Title":" Google Software Engineer","Description": "Learn to Code" ,"skills":["Java", "Python", "CSS"] }'
    totalScore = 0
    # textcv = json.loads(resume)
    # textjd = json.loads(jd)
    #print(skills_resume)
    #print(skills)
    documents = [skills_resume, skills]
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
    return ("Your resume matched " + str(answer) + " %" + " of the job-description!")
    #tokens = nlp("Java Python")
    #token1, token2 = tokens[0], tokens[1]

    '''
    for i in resume["skills"]:
        for j in jobDescription:
            token1 = nlp(i)
            token2 = nlp(j)
            print("Similarity of: ", i, "and", j, "is", token1.similarity(token2))
            totalScore = totalScore + token1.similarity(token2)
    print("Total Score Is: ", totalScore)
    #print("Similarity:", token1.similarity(token2))
    #print(resume["skills"][1])

    return jsonify(skills), 200
    '''
'''
{"_id":{"$oid":"62474043d46b2ee8dac4bba1"},"userID":"60cfc4d9d48268570cf3f07a","__v":{"$numberInt":"0"},"connections":[],"createdAt":{"$date":{"$numberLong":"1648836675316"}},"education":[{"university":"University of Massachusetts, Amherst","degree":"Bachelor of Science in Computer Science, Statistics & Data Science","fieldofstudy":[],"gpa":"3.835","from":"","to":"05/2022","current":false,"description":"GPA: 3.835\nRelevant Coursework: Algorithms, Data Structures, Software Engineering, Scalable Web Systems, Computer Network Theory,\nDatabases, Information Retrieval, Data Science, Natural Language Processing, Artificial Intelligence\nAmherst, MA\n","_id":{"$oid":"62474043238d52195c1cf3a6"}}],"experience":[{"title":"Software Engineering Intern","company":"MathWorks","from":"01/2021","to":"12/2021","current":false,"description":"● Modernizing & Developing JavaScript Web Components for MathWorks products with HTML, CSS, JS & React.\n","_id":{"$oid":"62474043238d52195c1cf3a1"}},{"title":"Software Engineering Intern","company":"Dell Technologies","from":"05/2021","to":"08/2021","current":false,"description":"● Collaborated to prototype edge-computing platforms with Litmus Edge and VMware to optimize the manufacturing industry.\n● Capture, analyze, visualize, and manage industrial data with Python, Tensorflow, JS, Docker, Postgres, TimescaleDB, and Grafana.\n● Innovated ML/Time Series models such as Kalman Filters and SARIMAX to estimate noisy trends from industrial MQTT sensors.\n● Monitored and performance tested Litmus Edge within VMware, supporting up to 10,000 industrial automation devices.\n","_id":{"$oid":"62474043238d52195c1cf3a2"}},{"title":"Software Engineering Intern","company":"Systems & Technology Research","from":"05/2020","to":"08/2020","current":false,"description":"● Designed a Spring Boot/Java REST service to rapidly automate assurance cases & causal models for predictive maintenance.\n● Developed Bayesian Networks to support causal model architecture and Uber Model Manager to manage different models.\nBUILD UMass\n","_id":{"$oid":"62474043238d52195c1cf3a3"}},{"title":"Software Engineer","company":"Systems & Technology Research","from":"01/2019","to":"Present","current":false,"description":"Github: https://git.io/JLpVa\n● Provide non-profits, startups, and local businesses with web and mobile applications through pro-bono engagements.\n● Grew organization to over 50 software developers, 20 business developers, and partnered with over 10 clients over two years.\n● Led two teams of 8 people with the cofounder of BUILD UMass to develop an auth/user management & forms/surveys system\n● Developing JSON form-building application to customize form-making, filling, as well as drag-and-drop capabilities.\n● Employed React for front-end UI, Node/Express.js/MongoDB for backend, along with Redux & Passport/JWT Integration.\nDSC-WAV (National Science Foundation)\n","_id":{"$oid":"62474043238d52195c1cf3a4"}},{"title":"Data Scientist","company":"Systems & Technology Research","from":"01/2020","to":"01/2021","current":false,"description":"Github: https://git.io/JLpah\n● Research to provide better transportation for people with drug addiction to get to treatment centers on an eight-person team.\n● Developed visualizations and analyzed the cost-time benefit to get to treatment centers with Python and Google Cloud.\n","_id":{"$oid":"62474043238d52195c1cf3a5"}}],"followers":[],"following":[],"invitationReceived":[],"invitationSent":[],"name":"Timothy Nguyen","numOfPosts":{"$numberInt":"0"},"phone":"6178883076","skills":["Prototype","R","C","Github","Architecture","Analytics","Design","Vmware","Docker","Cloud","Sql","Spark","Numpy","Statistics","Jira","Computer science","Engineering","User stories","Linux","Tensorflow","System","Transportation","Software engineering","Pandas","Mobile","Programming","Rest","Html","Python","Java","Algorithms","Json","Ui","Research","Automation","Js","Analyze","Css","Javascript"],"views":[]}
'''
if __name__ == '__main__':
    #define the localhost ip and the port that is going to be used
    # in some future article, we are going to use an env variable instead a hardcoded port 
    app.run(host='0.0.0.0', port=5002)

    