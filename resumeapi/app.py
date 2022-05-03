import os
from flask import Flask, request, jsonify
import io
import os
from dotenv import load_dotenv
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('maxent_ne_chunker')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('brown')
import spacy
import pandas as pd
import docx2txt
import constants as cs
from spacy.matcher import Matcher
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import requests
import phonenumbers
from date_extractor import extract_dates
import constants as cs
from job_titles.src.find_job_titles import FinderAcora
# import en_core_web_sm
from werkzeug.utils import secure_filename
import json
import logging
import fitz#c
import boto3
import uuid
from urllib.request import Request, urlopen
# load default skills data base
from skillNer.general_params import SKILL_DB
# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor
from spacy.matcher import PhraseMatcher
from flask_cors import CORS, cross_origin

load_dotenv()

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# load pre-trained model
nlp = spacy.load('en_core_web_sm')
finder=FinderAcora()
# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)
s3 = boto3.client('s3', region_name=os.getenv('REGION'), 
                aws_access_key_id=os.getenv('AWS_KEY_ID'), 
                aws_secret_access_key=os.getenv('AWS_SECRET_KEY'))


def my_replace(match):
    match = match.group()
    return match[0] + (" " if " " in match else "")
regex = r"[!\"#$%&\'()*+,\-.\/:;<=>?@\[\\\]^_`{|}~ ]{2,}"

def get_company_list():
    s3_file_key = "companies_sorted.csv"
    obj = s3.get_object(Bucket=os.getenv('BUCKET'), Key=s3_file_key)
    reader = pd.read_csv(obj['Body'])
    companies_word_list = []
    for row in reader['name']:
        companies_word_list.append(row)
    return companies_word_list
company_list = get_company_list()

def extract_text_from_pdf(filename):
    """
    A utility function to convert a machine-readable PDF to raw text.

    This code is largely borrowed from existing solutions, and does not match the style of the rest of this repo.
    :param input_pdf_path: Path to the .pdf file which should be converted
    :type input_pdf_path: str
    :return: The text contents of the pdf
    :rtype: str
    """
    # https://www.blog.pythonlibrary.org/2018/05/03/exporting-data-from-pdfs-with-python/
    # try:
    #     raw_text = parser.from_file(pdf_path,service='text')['content']
        
    #     full_string = re.sub(r'\n+','\n',raw_text)
    #     full_string = full_string.replace("\r", "\n")
    #     full_string = full_string.replace("\t", " ")

    #     # Remove awkward LaTeX bullet characters
        
    #     full_string = re.sub(r"\uf0b7", " ", full_string)
    #     full_string = re.sub(r"\u200b", " ", full_string)
    #     full_string = re.sub(r"\(cid:\d{0,2}\)", " ", full_string)
    #     # full_string = re.sub(r'•', " ", full_string)
    #     # full_string = re.sub(r'●', " ", full_string)

    #     # Split text blob into individual lines
    #     resume_lines = full_string.splitlines(True)
    #     return resume_lines

    # except Exception as e:
    #     print('Error in pdf file:: ' + str(e))
    #     return [], " "
    """
    A utility function to convert a machine-readable PDF to raw text.

    This code is largely borrowed from existing solutions, and does not match the style of the rest of this repo.
    :param input_pdf_path: Path to the .pdf file which should be converted
    :type input_pdf_path: str
    :return: The text contents of the pdf
    :rtype: str
    """
    s3_file_key = filename
    obj = s3.get_object(Bucket=os.getenv('BUCKET'), Key=s3_file_key)    
    fs = obj['Body'].read()
    try:
        for page in PDFPage.get_pages(
                        io.BytesIO(fs),
                        caching=True,
                        check_extractable=True
        ):
            resource_manager = PDFResourceManager()
            fake_file_handle = io.StringIO()
            converter = TextConverter(
                resource_manager,
                fake_file_handle,
                codec='utf-8',
                laparams=LAParams()
            )
            page_interpreter = PDFPageInterpreter(
                resource_manager,
                converter
            )
            page_interpreter.process_page(page)

            raw_text = fake_file_handle.getvalue()
            full_string = re.sub(r'\n+','\n',raw_text)
            full_string = full_string.replace("\r", "\n")
            full_string = full_string.replace("\t", "\n")
            full_string = re.sub(r'[ \t]{3,}', '\n', full_string)

            # Remove awkward LaTeX bullet characters

#                     full_string = re.sub(r"\uf0b7", " ", full_string)
#                     full_string = re.sub(r"\u200b", " ", full_string)
#                     full_string = re.sub(r"\(cid:\d{0,2}\)", " ", full_string)
#                     full_string = re.sub(r'•', " ", full_string)
#                     full_string = re.sub(r'\xa0', " ", full_string)
#                     full_string = re.sub(r'●', " ", full_string)
            yield full_string

            # close open handles
            converter.close()
            fake_file_handle.close()
    except PDFSyntaxError:
        return

def extract_text_from_doc(doc_path):
    '''
    Helper function to extract plain text from .doc or .docx files
    :param doc_path: path to .doc or .docx file to be extracted
    :return: string of extracted text
    '''
    try:
        raw_text = docx2txt.process(doc_path)
        full_string = re.sub(r'\n+','\n',raw_text)
        full_string = full_string.replace("\r", "\n")
        full_string = full_string.replace("\t", " ")

        # Remove awkward LaTeX bullet characters
        
        full_string = re.sub(r"\uf0b7", " ", full_string)
        full_string = re.sub(r"\u200b", " ", full_string)
        full_string = re.sub(r"\(cid:\d{0,2}\)", " ", full_string)
        # full_string = re.sub(r'•', " ", full_string)
        # full_string = re.sub(r'●', " ", full_string)
        
        
        # Split text blob into individual lines
        resume_lines = full_string.splitlines(True)
        return resume_lines

    except KeyError:
        text = textract.process(doc_path)
        text = text.decode("utf-8")
        clean_text = text.replace("\r", "\n").replace("\t", " ")  # Normalize text blob
        resume_lines = clean_text.splitlines(True)  # Split text blob into individual lines
        resume_lines = [re.sub('\s+', ' ', line.strip()) for line in resume_lines if line.strip()]  # Remove empty strings and whitespaces
        return ' '.join(resume_lines)

# https://towardsdatascience.com/something-from-nothing-use-nlp-and-ml-to-extract-and-structure-web-data-3f49b2f72b13
def extract_name(resume_text):
    #Using NER to find people names in the text. 
    doc=nlp(resume_text)
    persons=[X.text for X in doc.ents if X.label_ == 'PERSON']
    persons_dict=dict.fromkeys(persons,0)
    persons=list(persons_dict)

    final_names=[]
    for person in persons: 
        if len(word_tokenize(person)) >= 2:
            string_name=re.sub(r"[^a-zA-Z0-9]+", ' ', person).strip()
            final_names.append(string_name)
    return final_names[0] if len(final_names) > 0 else ""

def extract_mobile_number(text):
    '''
    Helper function to extract mobile number from text
    :param text: plain text extracted from resume file
    :return: string of extracted mobile numbers
    '''
    # Found this complicated regex on : https://zapier.com/blog/extract-links-email-phone-regex/
    try:
        return list(iter(phonenumbers.PhoneNumberMatcher(text, None)))[0].raw_string
    except:
        try:
            phone = re.findall(re.compile(r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'), text)
            if phone:
                number = ''.join(phone[0])
                if len(number) > 10:
                    return '+' + number
                else:
                    return number
        except:
            return ""

def extract_mobile_number(text):
    '''
    Helper function to extract mobile number from text
    :param text: plain text extracted from resume file
    :return: string of extracted mobile numbers
    '''
    # Found this complicated regex on : https://zapier.com/blog/extract-links-email-phone-regex/
    try:
        return list(iter(phonenumbers.PhoneNumberMatcher(text, None)))[0].raw_string
    except:
        try:
            phone = re.findall(re.compile(r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'), text)
            if phone:
                number = ''.join(phone[0])
                if len(number) > 10:
                    return '+' + number
                else:
                    return number
        except:
            return ""

def extract_email(email):
    email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", email)
    if email:
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return None

skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

def extract_skills(resume_text):
    '''
    Helper function to extract skills from spacy nlp text
    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :param noun_chunks: noun chunks extracted from nlp text
    :return: list of skills extracted
    '''
    # noun_chunks = nlp.noun_chunks
    '''
    nlp_text = nlp(resume_text)
    
    # removing stopwords and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
    
    # reading the csv file
    # data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'skills.csv')) 
    data = pd.read_csv("skills.csv")
    
    # extract values
    skills = list(data.columns.values)

    skillset = set()
    
    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in skills:
            skillset.add(token)
    
    for token in nlp_text.noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.add(token)
    '''
    # init skill extractor

    # extract skills from job_description
    annotations = skill_extractor.annotate(resume_text)
    skills = annotations['results']['full_matches'] + annotations['results']['ngram_scored']
    skills = set(elem['doc_node_value'].lower() for elem in skills)     
    return [i.capitalize() for i in set([i.lower() for i in skills])]

def get_university_list():
    # file_name = "https://raw.githubusercontent.com/jineshdhruv8/ResumeParser/master/Code/ResumeParser/wordList/companies.csv"
    file_name = "degree.csv"
    reader = pd.read_csv(file_name)
    companies_word_list = []
    for row in reader:
        print(row)
        companies_word_list.append(row)
    return companies_word_list
    

def extract_sections(resume_text):
    RESUME_SECTIONS_GRAD = (
        'accomplishments',
        'experience',
        'education',
        'interests',
        'projects',
        'publications',
        'skills',
        'certifications',
        'summary'
    )

    total = cs.education_and_training + cs.work_and_employment + cs.project + RESUME_SECTIONS_GRAD
    text_split = [i.strip() for i in resume_text.split('\n')]
    entities = {}
    key = False
    for phrase in text_split:
        if len(phrase) == 1:
            p_key = phrase
        else:
            p_key = set(phrase.lower().split()) & set(total)
        try:
            p_key = list(p_key)[0]
        except IndexError:
            pass
        if p_key in total:
            if p_key in cs.project:
                key = 'project'
            elif p_key in cs.work_and_employment:
                key = 'experience'
            elif p_key in cs.education_and_training:
                key = 'education'
            else:
                key = p_key
            entities[key] = []
        elif key and phrase.strip():
            entities[key].append(phrase)
    return entities

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

def extract_university(input_text):
    organizations = []

    # we search for each bigram and trigram for reserved words
    # (college, university etc...)
    education = list()
    for word in RESERVED_WORDS:
        if input_text.lower().find(word) >= 0:
            education.append(input_text)
    
    return education

def extract_education_section(entities):
    university_dict = dict()
    nlp = spacy.load('en_core_web_sm')
    curr_university = ''
    curr_degree = ''
    begin_date = ''
    end_date = ''
    curr_gpa = ""
    university_dict[curr_university] = {"details": [["", "", "", "", ""]]}


    def get_gpa(text):
        rx = re.compile(r'(\bgpa[ :]+)?(\d+(?:\.\d+)?)[/\d. ]{0,6}(?(1)| *gpa\b)', re.I)
        m = rx.search(text)
        if m:
            return m.group(2)
        return ""
        # print(text, "=>", m.group(2), sep=" ")

    for text in entities['education']:
        found_date = False
        found_degree = False
        found_university = False
        university_list = extract_university(text)
        degree_list = extract_degree(text)
        # extracting entities

        if len(university_list) > 0:
            curr_university = university_list[0]
            if curr_university not in university_dict:
                university_dict[curr_university] = {"details": [["", "", "", "", ""]]}
            found_university = True
        if len(degree_list) > 0:
            curr_degree = ''
            curr_gpa = ""
            for degree in degree_list:
                curr_degree += degree
            university_dict[curr_university]['details'].append([curr_degree, begin_date, end_date, curr_gpa, ""])
            found_degree = True
        
        new_gpa = get_gpa(text)
        if curr_gpa == '' and new_gpa != '':
            curr_gpa = new_gpa
            university_dict[curr_university]["details"][-1][3] = curr_gpa
            
        if end_date == '' and '.' not in text:       
            dates = extract_dates(text)
            if dates:
                date_time_start = dates[0].strftime("%m/%Y") 
                date_time_end = dates[0].strftime("%m/%Y")
                if len(dates) > 1:
                    date_time_end = dates[1].strftime("%m/%Y")
                word_list = text.split()
                for word in word_list:
                    if word.lower() == 'now' or  word.lower() == 'present':
                        date_time_end = 'Present'
                # new_date = date_time_start + " - " + date_time_end
                begin_date = date_time_start 
                end_date = date_time_end
                if date_time_end == date_time_start:
                    begin_date = ''
                found_date = True
                university_dict[curr_university]['details'][-1][1] = begin_date
                university_dict[curr_university]['details'][-1][2] = end_date
        
        
        if len(university_dict[curr_university]['details']) > 0:
            university_dict[curr_university]['details'][-1][2] = end_date
        if not found_date and not found_degree and not found_university:
            university_dict[curr_university]["details"][-1][4] += (text + "\n")
    filtered_university = {comp:details for comp, details in university_dict.items() if len(details['details']) != 0}
    education = []
    for uni in filtered_university:
        for entry in filtered_university[uni]['details']:
            data = {}
            data['degree'] = entry[0]
            data['university'] = uni
            data['from'] = entry[1]
            data['to'] = entry[2]
            data['gpa'] = entry[3]
            data['description'] = entry[4]
            education.append(data)
    # json_data_education = json.dumps(education)
    return education

def extract_work_experience(entities):
    # company = dict()
    # # des_set = set(designation_list.lower())
    # nlp = spacy.load('en_core_web_sm')
    # curr_company = 'Unnamed'
    # company[curr_company] = {"date": [], "designation": []}
    # if 'experience' in entities:
    #     for text in entities['experience']:
    #         test_text = text.split()
    #         foundCompany = False
    #         if len(test_text) <= 6:
    #             for word in company_list:
    #                 if text.lower() == word:
    #                     if text not in company:
    #                         curr_company = text
    #                         company[text] = {"date": [], "designation": []}
    #                     break
    #         match = finder.findall(text)
    #         # merge intervals
    #         if len(match) > 0:
    #             s = [[match[0].start, match[0].end]]
    #             for i in range(1, len(match)):
    #                 if s[-1][1] >= match[i].start:
    #                     s[-1][1] = match[i].end
    #                 else:
    #                     s.append([match[i].start, match[i].end])
    #             if s:
    #                 for elem in s:
    #                     t = str(text[elem[0]:elem[1]])
    #                     company[curr_company]["designation"].append([t, ""])
                        
    #         text1 = nlp(text)
    #         found_date = False
    #         for word in text1.ents:
    #             if word.label_ == 'DATE':
    #                 date = text1[word.start:word.end]
    #                 company[curr_company]["date"].append(str(date))
    #                 found_date = True
    #                 break
    #         if curr_company and not foundCompany and not found_date and len(match) == 0:
    #             if len(company[curr_company]["designation"]) > 0:
    #                 company[curr_company]["designation"][-1][1] += (text + "\n")
    # return company
    finder=FinderAcora()
    company = dict()
    designation = list()
    # des_set = set(designation_list.lower())
    nlp = spacy.load('en_core_web_sm')
    curr_company = ''
    begin_date = ''
    end_date = ''
    company[curr_company] = {"designation": [["", "", "", ""]]}
    for text in entities['experience']:
        test_text = text.split(r'\s*[-,;|]\s+')
        foundCompany = False
        if len(test_text) <= 6:
            # temp_text= text.split(delim = re.compile(r'\s*[-,;]\s+'))
            for word in company_list:
                if word == text.lower():
                    begin_date = ""
                    end_date = ""
                    if text not in company:
                        curr_company = text
                        company[text] = {"designation": [["", "", "", ""]]}
                    foundCompany = True
        match = finder.findall(text)
        # merge intervals
        if len(match) > 0:
            s = [[match[0].start, match[0].end]]
            for i in range(1, len(match)):
                if s[-1][1] >= match[i].start:
                    s[-1][1] = match[i].end
                else:
                    s.append([match[i].start, match[i].end])
            if s:
                for elem in s:
                    t = text[elem[0]:elem[1]]
                    company[curr_company]["designation"].append([begin_date, end_date, t, ""])
                    
        found_date = False
        if len(test_text) <= 8:
            dates = extract_dates(text)   
            if dates:
                date_time_start = dates[0].strftime("%m/%Y") 
                date_time_end = dates[0].strftime("%m/%Y")
                if len(dates) > 1:
                    date_time_end = dates[1].strftime("%m/%Y")
                word_list = text.split()
                for word in word_list:
                    if word.lower() == 'now' or  word.lower() == 'present':
                        date_time_end = 'Present'
                begin_date = date_time_start
                end_date = date_time_end
                found_date = True
        if not foundCompany and not found_date  and len(match) == 0:
            company[curr_company]["designation"][-1][3] += (text + "\n")
        if begin_date != '':
            company[curr_company]["designation"][-1][0] = begin_date
            company[curr_company]["designation"][-1][1] = end_date
    filtered_company = {comp:value for comp, value in company.items() if len(value['designation']) != 0}
    experience = []
    for company in filtered_company:
        for entry in filtered_company[company]['designation']:
            data = {}
            data['title'] = entry[2]
            data['company'] = company
            data['from'] = entry[0]
            data['to'] = entry[1]
            data['description'] = entry[3]
            experience.append(data)
    # json_data_experience = json.dumps(experience)
    return experience

ALLOWED_EXTENSIONS = set(['pdf'])

@app.route('/resumeapi', methods=['GET'])
@cross_origin()
def hello():
    return 'Hello World'

def upload_file(resume):
    # filename = secure_filename(resume.filename)
    filename = str(uuid.uuid4())
    resume.save(filename)
    s3.upload_file(
        Bucket = os.getenv('BUCKET'),
        Filename = filename, 
        Key = filename)
    return filename

#we define the route
@app.route('/resumeapi/parse_resume', methods=['POST'])
@cross_origin()
def parseResume():
    # return a json
    logging.info(request.files)
    resume = request.files['file']
    if resume:
        filename = upload_file(resume)
        '''
        doc = fitz.open(stream=resume.read(), filetype="pdf")
        resume_text = ""
        for page in doc.pages():
            raw_text = page.get_text("text")
            full_string = re.sub(r'\n+','\n',raw_text)
            full_string = full_string.replace("\r", "\n")
            full_string = full_string.replace("\t", "\n")
            full_string = re.sub(r'[ \t]{3,}', '\n', full_string)
            resume_text += full_string
        print(resume_text)
        '''
        # filename = secure_filename(resume.filename)
        resume_text = ""
        for page in extract_text_from_pdf(filename):
            resume_text += ' ' + page
        print(resume_text)
        # Make sure to delete filename here
        s3.delete_object(Bucket = os.getenv('BUCKET'), Key = filename)
        # Extract names & other info
        name = extract_name(resume_text)
        phone = extract_mobile_number(resume_text)
        email = extract_email(resume_text)
        skills = extract_skills(resume_text)
        entities = extract_sections(resume_text)
        experience_section = {}
        education_section = {}
        if "experience" in entities:
            experience_section = extract_work_experience(entities)
        if "education" in entities:
            education_section = extract_education_section(entities)
        # education_degrees = extract_degree("\n".join(entities['education']))
        # universites = extract_education(entities['education'])
        res = {
            "name": name,
            "phone": phone,
            "email": email,
            "skills": skills,
            "experience": experience_section,
            "education": education_section
        }
        return res
    return {}
    # logging.info(resume_text)
    # return resume_text
    
@app.route('/', methods=['GET'])
def hi():
    return jsonify("Hello World")

if __name__ == '__main__':
    #define the localhost ip and the port that is going to be used
    # in some future article, we are going to use an env variable instead a hardcoded port 
    app.run(host='0.0.0.0', port=5001)