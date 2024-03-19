from pydantic import BaseModel

from langchain_community.llms import Bedrock
from langchain_community.llms import BedrockChat
from langchain.prompts import PromptTemplate

from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage



import boto3
# import json
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
def get_model(model_name = "meta.llama2-13b-chat-v1",system=None,history=None):
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
    elif model_name == 'anthropic.claude-3-sonnet-20240229-v1:0' or model_name == 'anthropic.claude-v2:1':
        body = {
                "max_tokens": 1000,
                # "system" : system,
                # "messages" : history,
                "temperature": 0.15,
                "top_p": 0.96,
                "stop_sequences" : ["[INST]","[User]","[Assistant]","[\n]","[ ]","[\n ]","[ \n]","[Customer","Bot:","Human:"]
            }
        return BedrockChat(client = bedrock, model_id=model_name,model_kwargs=body)


def claude_message_api_formatter(system_message,message_history):
    messages = []
    messages.append(SystemMessage(content=system_message))
    for msg in message_history:
        # messages.append({"role" : "user","content" : msg["user"]})
        # messages.append({"role" : "assistant","content" : msg["chatbot"]})
        messages.append(HumanMessage(content=msg["user"]))
        messages.append(AIMessage(content=msg["chatbot"]))
    return messages

# define the retriever and prompt
retriever = index.vectorstore.as_retriever()
template: str = '''[INST]\n{context} [/INST]\n{history} \n[INST]\n{question} [/INST]'''
prompt = PromptTemplate.from_template(template=template)

@app.get("/api/test")
async def test(request: Request):
    return "Hello world!"

# main route
@app.post("/chat", response_class=StreamingResponse)
async def chat(request: Request):
    # get request variables and print them out
    request_data = await request.json()
    
    print(request_data)
    message = request_data["message"]
    history = request_data["history"]
    model = request_data["model"]

    # get relevant documents
    docs = retriever.get_relevant_documents(message)

    # define the system prompt
    system = f'''
    Context: {str(docs[0].page_content) + ' ' + str(docs[1].page_content) + ' ' + str(docs[2].page_content)}
    You are an AI chatbot for the RIDE, an MBTA paratransit service. You will help customer service representatives respond to user complaints and queries.
    Answer questions based on your knowledge and nothing more. If you are unable to decisively answer a question, direct them to customer service. Do not make up information outside of your given information.
    Customer service is needed if it is something you cannot answer. Requests for fare history require customer service, as do service complaints like a rude driver or late pickup.
    Highly-specific situations will also require customer service to step in.
    Phone numbers:
    TRAC (handles scheduling/booking, trip changes/cancellations, anything time-sensitive): 844-427-7433 (voice/relay) 857-206-6569 (TTY)
    Mobility Center (handles eligibility questions, renewals, and changes to mobility status): 617-337-2727 (voice/relay)
    MBTA Customer support (handles all other queries): 617-222-3200 (voice/relay)
    '''

    # get the correct model
    llm = get_model(model)
    print(model)

    if "claude" in model:
        # if it's a Claude model, use the Message API - there's no reason to use the completion API in this case
        messages = claude_message_api_formatter(history)
        # messages.append({"role" : "user","content" : message})
        messages.append(HumanMessage(content=message))
        prompt_to_send = messages
    else:
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
        # a bit of a hack - prompt_to_send could either be a list of MessageAPI messages or a simple string
        # and hopefully the earlier code prevents any mismatches
        async for chunk in llm.astream(prompt_to_send):
            # print(chunk.__anext__())
            yield chunk #.__anext__()


    return StreamingResponse(stream_response(), media_type="application/json")
