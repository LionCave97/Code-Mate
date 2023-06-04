from fastapi import FastAPI
import tempfile
import subprocess
import traceback
import sys
import os
from io import StringIO
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
import langchain.agents as lc_agents
from langchain.llms import OpenAI
import logging
from datetime import datetime
from langchain.llms import OpenAI as LangChainOpenAI
import openai
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"


global generated_code

LANGUAGE_CODES = {
    # 'C': 'c',
    'C++': 'cpp',
    # 'Java': 'java',
    # 'Ruby': 'ruby',
    # 'Scala': 'scala',
    'C#': 'csharp',
    'Objective C': 'objc',
    'Swift': 'swift',
    'JavaScript': 'nodejs',
    'Kotlin': 'kotlin',
    'Python': 'python3',
    # 'GO Lang': 'go',
}

@app.get("/hello")
def hello():
  return {"Hello world!"}

@app.get("/getlanguages")
def getLaunguages():
  return LANGUAGE_CODES



code_prompt = ""
code_language = ""

@app.get("/generateCode", response_class=HTMLResponse)
def generateCode(prompt, language):
    print("Data Received", prompt, language)
    global code_prompt
    global code_language
    print("Language API", language)
    # if prompt is None:
    #    prompt = "get the weather for pretoria"
    # if language is None:
    #    language = 'python'
    code_prompt = prompt
    code_language = language
    print("API Call", code_prompt, code_language)
    gencode = ""
    gencode =  generate_code()
    print("Code:", gencode)
    return gencode


# LLM Chains definition
# Create an OpenAI LLM model
open_ai_llm = OpenAI(temperature=0.7, max_tokens=1000)






# Memory for the conversation
memory = ConversationBufferMemory(
    input_key='code_topic', memory_key='chat_history')
# Create a chain that generates the code


# LLM Chains definition
# Create an OpenAI LLM model
open_ai_llm = OpenAI(temperature=0.7, max_tokens=1000)



def generate_code():
    logger = logging.getLogger(__name__)
    try:
        # print("Chain", code_chain)
        # Prompt Templates
        code_template = PromptTemplate(input_variables=['code_topic'], template='Write me code in ' + f'{code_language} language' + ' for {code_topic}')
        code_chain = LLMChain(llm=open_ai_llm, prompt=code_template, output_key='code', memory=memory, verbose=True)
        # print("Prompt", code_prompt)
        # print("Language", code_language)

        generatedCode = code_chain.run(code_prompt)
        codeLanguage = code_language
        # print(generatedCode)
        return(generatedCode)

    except Exception as e:
        
        logger.error(f"Error in code generation: {traceback.format_exc()}")