from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI
import requests
import os
import streamlit as st
import torch
import soundfile as sf

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUBHUB_API_TOKEN")

#img2text
def img2text(url):
    image_to_text = pipeline("image-to-text", model = "Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]["generated_text"]

    print(text)
    return text

#llm
def generate_story(scenario):
    template="""
    You are a storyteller;
    You can generate an advertising video script based on a simple narrative, divided into paragraphs, including storyboards, duration and other elements. The story should be around 800 words. Be specific when describing the scenario. The goal is to promote the product.
    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template,input_variables=["scenario"])

    story_llm=LLMChain(llm=ChatOpenAI(
            model_name="gpt-3.5-turbo",temperature=1),prompt=prompt, verbose=True)
    
    story=story_llm.predict(scenario=scenario, max_new_tokens=5000)
    print(story)
    return story



def main():
    st.set_page_config(page_title = "AI story Teller", page_icon ="🤖")

    st.header("We turn images to story!")
    upload_file = st.file_uploader("Choose an image...", type = 'jpg')  #uploads image

    if upload_file is not None:
        print(upload_file)
        binary_data = upload_file.getvalue()
        
        # save image
        with open (upload_file.name, 'wb') as f:
            f.write(binary_data)
        st.image(upload_file, caption = "Image Uploaded", use_column_width = True) # display image

        scenario = img2text(upload_file.name) #text2image
        story = generate_story(scenario) # create a story

        # display scenario and story
        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        
    

# the main
if __name__ == "__main__":
    main()

