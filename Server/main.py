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
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# Start Server:
#  uvicorn main:app --reload
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




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
add_prompt = None
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

@app.post("/addCode", response_class=HTMLResponse)
def addCode(prompt, add, language):
    print("Data Received", prompt, language)
    global code_prompt
    global code_language
    global add_prompt
    print("Add", add)

    # addReceived = Request(add)
    print("Language API", language)
    # print("Add", addReceived)

    # if prompt is None:
    #    prompt = "get the weather for pretoria"
    # if language is None:
    #    language = 'python'
    code_prompt = prompt
    add_prompt = add
    code_language = language
    print("API Call", code_prompt, code_language)
    gencode = ""
    gencode =  add_code()
    print("Code:", gencode)
    return gencode

@app.post("/whatCode", response_class=HTMLResponse)
def addCode(prompt):
    print("Data Received", prompt)
    global code_prompt

    # if prompt is None:
    #    prompt = "get the weather for pretoria"
    # if language is None:
    #    language = 'python'
    code_prompt = prompt
    print("API Call", code_prompt)
    gencode = ""
    gencode =  what_code()
    print("Code:", gencode)
    return gencode

@app.post("/improve", response_class=HTMLResponse)
def addCode(prompt, advice):
    print("Data Received", prompt)
    global code_prompt
    global add_prompt
    print("Add", advice)

    # addReceived = Request(add)
    # print("Add", addReceived)

    # if prompt is None:
    #    prompt = "get the weather for pretoria"
    # if language is None:
    #    language = 'python'
    code_prompt = prompt
    add_prompt = advice
    print("API Call", code_prompt)
    gencode = ""
    gencode =  improve_code()
    print("Code:", gencode)
    return gencode


# LLM Chains definition
# Create an OpenAI LLM model
open_ai_llm = OpenAI(temperature=0.7, max_tokens=1000)


# Memory for the conversation
memory = ConversationBufferMemory(
    input_key='code_topic', memory_key='chat_history')
# Create a chain that generates the code



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

def add_code():
    logger = logging.getLogger(__name__)
    try:
        # print("Chain", code_chain)
        # Prompt Templates
        print("Add Prompt ", add_prompt)
        code_template = PromptTemplate(input_variables=['code_topic'], template=f'Take the existing code {add_prompt}' +' written in '+ f'{code_language} language' + ' and add the following {code_topic}')
        code_chain = LLMChain(llm=open_ai_llm, prompt=code_template, output_key='code', memory=memory, verbose=True)
        # print("Prompt", code_prompt)
        # print("add Prompt", add_prompt)

        # print("Language", code_language)

        addCode = code_chain.run(code_prompt)
        codeLanguage = code_language
        # print(generatedCode)
        return(addCode)

    except Exception as e:
        
        logger.error(f"Error in code generation: {traceback.format_exc()}")

def what_code():
    logger = logging.getLogger(__name__)
    try:
        # print("Chain", code_chain)
        # Prompt Templates
        print("Add Prompt ", add_prompt)
        code_template = PromptTemplate(input_variables=['code_topic'], template= 'what does this code do {code_topic} and in what language is it written?')
        code_chain = LLMChain(llm=open_ai_llm, prompt=code_template, output_key='code', memory=memory, verbose=True)
        # print("Prompt", code_prompt)
        # print("add Prompt", add_prompt)

        # print("Language", code_language)

        whatCode = code_chain.run(code_prompt)
        codeLanguage = code_language
        # print(generatedCode)
        return(whatCode)

    except Exception as e:
        
        logger.error(f"Error in code generation: {traceback.format_exc()}")

def improve_code():
    logger = logging.getLogger(__name__)
    try:
        print("Add Prompt ", add_prompt)
        code_template = PromptTemplate(input_variables=['code_topic'], template= ' give me advice on {code_topic} for this code'+ f' {add_prompt}' )
        code_chain = LLMChain(llm=open_ai_llm, prompt=code_template, output_key='code', memory=memory, verbose=True)

        adviceCode = code_chain.run(code_prompt)
        # print(generatedCode)
        return(adviceCode)

    except Exception as e:
        
        logger.error(f"Error in code generation: {traceback.format_exc()}")