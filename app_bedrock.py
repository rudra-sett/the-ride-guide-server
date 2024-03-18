from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory

from langchain.chains import LLMChain
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate

import boto3
import json
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse
import asyncio
from fastapi.middleware.cors import CORSMiddleware


bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
# model_id = 'mistral.mistral-7b-instruct-v0:2'
accept = 'application/json'
contentType = 'application/json'

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:3001",
    # Add any other origins you want to allow
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

def get_index(): #creates and returns an in-memory vector store to be used in the application
    
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
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class LlamaLLM(LLM):

    model_id = 'meta.llama2-13b-chat-v1'

    @property
    def _llm_type(self) -> str:
        return "Llama2 13B"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        body = json.dumps({
            "prompt": prompt,
            "max_gen_len": 1000,
            "temperature": 0.07,
            "top_p": 0.92,
        })
        response = bedrock.invoke_model(body=body,
                                            modelId=self.model_id,
                                            accept=accept,
                                                contentType=contentType)
        return json.loads(response.get('body').read())['generation'] 

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": "LLaMa 2 13B"}

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        body = json.dumps({
            "prompt": prompt,
            "max_gen_len": 1000,
            "temperature": 0.15,
            "top_p": 0.92,
        })
        response = bedrock.invoke_model_with_response_stream(body=body, modelId=self.model_id, accept=accept, contentType=contentType)

        # def stream_response():
        for event in response["body"]:
            if "chunk" in event:
                thing = json.loads(event['chunk'].get('bytes').decode())
                # print(thing)
                yield event['chunk'].get('bytes') #thing['generation']

        # return stream_response()
    
    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        body = json.dumps({
            "prompt": prompt,
            "max_gen_len": 1000,
            "temperature": 0.15,
            "top_p": 0.92,
        })
        print("hi")
        response = bedrock.invoke_model_with_response_stream(body=body, modelId=self.model_id, accept=accept, contentType=contentType)

        async def stream_wrapper(event_stream):
            # Assuming event_stream is the EventStream object you're dealing with
            # You need to implement fetching and yielding logic here
            # This is a placeholder implementation; adapt it to how your EventStream works
            async for event in event_stream:
                if "chunk" in event:
                    yield event['chunk'].get('bytes').decode()

        # Here we pass the EventStream (response["body"]) to the wrapper
        return stream_wrapper(response["body"])

        # async def stream_response():
        #     async for event in response["body"]:
        #         if "chunk" in event:
        #             # thing = await json.loads(event['chunk'].get('bytes').decode())
        #             # print(thing)
        #             yield event['chunk'].get('bytes').decode() #thing['generation']

        # return stream_response()

class MistralLLM(LLM):

    model_id = 'mistral.mistral-7b-instruct-v0:2'

    @property
    def _llm_type(self) -> str:
        return "Mistral 7B"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        body = json.dumps({
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.07,
            "top_p": 0.92,
            "stop" : ["[INST]"]
        })
        response = bedrock.invoke_model(body=body,
                                            modelId=self.model_id,
                                            accept=accept,
                                                contentType=contentType)
        return json.loads(response.get('body').read())['generation'] 

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": "Mistral 7B"}

    async def astream(self, prompt: str, **kwargs) -> Iterator[str]:
        body = json.dumps({
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.07,
            "top_p": 0.92,
            "stop" : ["[INST]"]
        })
        print("hi")
        response = bedrock.invoke_model_with_response_stream(body=body, modelId=self.model_id, accept=accept, contentType=contentType)

        async def stream_response():
            async for event in response["body"]:
                if "chunk" in event:
                    thing = json.loads(event['chunk'].get('bytes').decode())
                    # print(thing)
                    yield event['chunk'].get('bytes') #thing['generation']

        return stream_response()

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []

class ChatResponse(BaseModel):
    response: str


# llm = MistralLLM()
# llm = LlamaLLM()
def get_model(model_name = "meta.llama2-13b-chat-v1"):
    if model_name == "meta.llama2-13b-chat-v1":
        body = json.dumps({
                "max_gen_len": 1000,
                "temperature": 0.15,
                "top_p": 0.92,
            })
        return Bedrock(client = bedrock, model_id="meta.llama2-13b-chat-v1",model_kwargs=body)
    elif model_name == "meta.llama2-70b-chat-v1":
        body = json.dumps({
                "max_gen_len": 1000,
                "temperature": 0.15,
                "top_p": 0.92,
            })
        return Bedrock(client = bedrock, model_id="meta.llama2-70b-chat-v1",model_kwargs=body)
    elif model_name == "mistral.mistral-7b-instruct-v0:2":
        body = json.dumps({
                "max_tokens": 1000,
                "temperature": 0.15,
                "top_p": 0.96,
                "stop" : ["[INST]","[User]","[Assistant]","[\n]","[ ]","[\n ]","[ \n]","[Customer","Bot:","Human:"]
            })
        return Bedrock(client = bedrock, model_id="mistral.mistral-7b-instruct-v0:2",model_kwargs=body)

retriever = index.vectorstore.as_retriever()
template: str = '''[INST]\n{context} [/INST]\n{history} \n[INST]\n{question} [/INST]'''
prompt = PromptTemplate.from_template(template=template)

@app.post("/chat", response_class=StreamingResponse)
async def chat(request: Request):
    request_data = await request.json()
    # print(request_data)
    message = request_data["message"]
    history = request_data["history"]
    model = request_data["model"]
    print(model)
    llm = get_model(model)

    docs = retriever.get_relevant_documents(message)
    system = f'''
    Context: {str(docs[0].page_content) + ' ' + str(docs[1].page_content) + ' ' + str(docs[2].page_content)}
    You are an AI chatbot for the RIDE, an MBTA paratransit service. You will respond to user questions and complaints.
    Answer questions based on your knowledge and nothing more. If you are unable to decisively answer a question, direct them to customer service. Do not make up information outside of your given information.
    Customer service is needed if it is something you cannot answer. Requests for fare history require customer service, as do service complaints like a rude driver or late pickup.
    Highly-specific situations will also require customer service to step in.'''
    # Provide your output in this format, where you use JSON to provide a response to the user and a boolean flag
    # indicating if the query needs further support: 
    # {{"response to user" : <<your answer>>,
    # "customer_service_needed" : <<True or False>>}} ONLY USE THIS FORMAT
    # history = request.history 
    # question = request.message

    history_str = ""
    for msg in history:
        history_str += f'[INST]\n{msg["user"]} [/INST]\n' 
        history_str += f'{msg["chatbot"]}\n'

    prompt_to_send = prompt.format(context=system,question=message,history=history_str)
    print(prompt_to_send)
    # response = llm.invoke(prompt_to_send)
    
    # response = response.strip("<|im_end|>")
    # return ChatResponse(response=json.dumps({"response_to_user" : response}))

    async def stream_response():
        nonlocal prompt_to_send
        async for chunk in llm.astream(prompt_to_send):
            # print(chunk.__anext__())
            yield chunk #.__anext__()


    return StreamingResponse(stream_response(), media_type="application/json")
