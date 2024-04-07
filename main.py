import gradio as gr
import random
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import openai

def random_response(message, history):
    return random.choice(["Yes", "No"])

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


llm = ChatOpenAI(temperature=0.15, model='gpt-3.5-turbo-0613')

def predict(message, history):
    history_langchain_format = [HumanMessage(role="system", content="You are a therapist. Explore with the patient to understand why they feel certain emotions, and keep the conversation going with the patient, constantly trying to understand why they feel a certain way. After appropriate exploring, offer solutions, and then keep the conversation going.")]
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    print(history_langchain_format)
    gpt_response = llm(history_langchain_format)
    return gpt_response.content

gr.ChatInterface(predict).launch()

# from datasets import load_dataset
#
# dataset = load_dataset("nbertagnolli/counsel-chat")