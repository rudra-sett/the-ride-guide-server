from pydantic import BaseModel

from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate

import boto3
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
accept = 'application/json'
contentType = 'application/json'

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://therideguidebucket.s3-website-us-east-1.amazonaws.com",
    "http://localhost:3000",
    "http://therideguide.us-east-1.elasticbeanstalk.com"    # Add any other origins you want to allow
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define the path to the pre-trained model you want to use
modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

#creates and returns an in-memory vector store to be used in the application
def get_index(): 
    
    loader = TextLoader("./The Ride Guide.txt")
    
    text_splitter = RecursiveCharacterTextSplitter( #create a text splitter
        separators=["\n\n", "\n", ".", " "], #split chunks at (1) paragraph, (2) line, (3) sentence, or (4) word, in that order
        chunk_size=1000, #divide into 1000-character chunks using the separators above
        chunk_overlap=100 #number of characters that can overlap with previous chunk
    )
    
    index_creator = VectorstoreIndexCreator( #create a vector store factory
        vectorstore_cls=FAISS, #use an in-memory vector store for demo purposes
        embedding=embeddings, #use Titan embeddings
        text_splitter=text_splitter, #use the recursive text splitter
    )
    
    index_from_loader = index_creator.from_loaders([loader]) #create an vector store index from the loaded PDF
    
    return index_from_loader #return the index to be cached by the client app

index = get_index()

from typing import Any, AsyncIterator, Iterator, List, Mapping, Optional

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []

class ChatResponse(BaseModel):
    response: str

# get the correct model and params
def get_model(model_name = "meta.llama2-13b-chat-v1"):
    if model_name == "meta.llama2-13b-chat-v1":
        body = {
                "max_gen_len": 1000,
                "temperature": 0.15,
                "top_p": 0.92,
            }
        return Bedrock(client = bedrock, model_id="meta.llama2-13b-chat-v1",model_kwargs=body)
    elif model_name == "meta.llama2-70b-chat-v1":
        body = {
                "max_gen_len": 1000,
                "temperature": 0.15,
                "top_p": 0.92,
            }
        return Bedrock(client = bedrock, model_id="meta.llama2-70b-chat-v1",model_kwargs=body)
    elif model_name == "mistral.mistral-7b-instruct-v0:2":
        body = {
                "max_tokens": 1000,
                "temperature": 0.15,
                "top_p": 0.96,
                "stop" : ["[INST]","[User]","[Assistant]","[\n]","[ ]","[\n ]","[ \n]","[Customer","Bot:","Human:"]
            }
        return Bedrock(client = bedrock, model_id="mistral.mistral-7b-instruct-v0:2",model_kwargs=body)

# define the retriever and prompt
retriever = index.vectorstore.as_retriever()
template: str = '''[INST]\n{context} [/INST]\n{history} \n[INST]\n{question} [/INST]'''
prompt = PromptTemplate.from_template(template=template)

# main route
@app.post("/chat", response_class=StreamingResponse)
async def chat(request: Request):
    # get request variables and print them out
    request_data = await request.json()
    
    print(request_data)
    message = request_data["message"]
    history = request_data["history"]
    model = request_data["model"]

    # get the correct model
    print(model)
    llm = get_model(model)

    # get relevant documents
    docs = retriever.get_relevant_documents(message)

    # define the system prompt
    system = f'''
    Context: {str(docs[0].page_content) + ' ' + str(docs[1].page_content) + ' ' + str(docs[2].page_content)}
    You are an AI chatbot for the RIDE, an MBTA paratransit service. You will help customer service representatives respond to user complaints and queries.
    Answer questions based on your knowledge and nothing more. If you are unable to decisively answer a question, direct them to customer service. Do not make up information outside of your given information.
    Customer service is needed if it is something you cannot answer. Requests for fare history require customer service, as do service complaints like a rude driver or late pickup.
    Highly-specific situations will also require customer service to step in.'''

    # assemble the chat history
    history_str = ""
    for msg in history:
        history_str += f'[INST]\n{msg["user"]} [/INST]\n' 
        history_str += f'{msg["chatbot"]}\n'

    # assemble the prompt
    prompt_to_send = prompt.format(context=system,question=message,history=history_str)
    print(prompt_to_send)

    # return a streamed response
    async def stream_response():
        nonlocal prompt_to_send
        async for chunk in llm.astream(prompt_to_send):
            # print(chunk.__anext__())
            yield chunk #.__anext__()


    return StreamingResponse(stream_response(), media_type="application/json")
