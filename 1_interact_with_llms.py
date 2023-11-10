import os
import configparser

from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain

# Create a ConfigParser object and read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# CONSTANTS
HUGGINGFACEHUB_API_TOKEN = config.get('huggingface', 'HUGGINGFACE_API_TOKEN')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
repo_id = "google/flan-t5-xxl" 

# Prompt Building
template = """Question: {question}

Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])

# API inference endpoint
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.2, "max_length": 64}
)

# Create Chain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# For single question
question = "Where is the Sky Needle Located?"
print(llm_chain.run(question))

# For multiple questions
question_list = [
    {'question':'Current year is 2023. I was born in 2000. How old am I?'},
    {'question':'Convert 85 fahrenheit to celsius'},
    {'question':'What is the capital of Maharashtra?'},
    {'question':'Do you like Smokers?'}
]
result = llm_chain.generate(question_list)
print(result)
