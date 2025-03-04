import streamlit as st
import requests
import warnings
warnings.filterwarnings('ignore')

import os
import yaml
import threading
import time
from crewai import Agent, Task, Crew
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")


# Define structured output schema
class ContentOutput(BaseModel):
    article: str = Field(..., description="The article, formatted in markdown.")

# Load YAML configurations for agents and tasks
files = {
    'agents': 'config/agents.yaml',
    'tasks': 'config/tasks.yaml'
}
configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r') as file:
        configs[config_type] = yaml.safe_load(file)

agents_config = configs['agents']
tasks_config = configs['tasks']

# Load tools
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# Creating Agents
article_fetcher = Agent(
    config=agents_config['article_fetcher'],
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
)

content_creator_agent = Agent(
    config=agents_config['podcast_creator'],
)

quality_assurance_agent = Agent(
    config=agents_config['quality_assurance_agent'],
)

# Creating Tasks
fetch_text = Task(
    config=tasks_config['get_the_text_of_url'],
    agent=article_fetcher
)

create_content_task = Task(
    config=tasks_config['create_content'],
    agent=content_creator_agent,
    context=[fetch_text]
)

quality_assurance_task = Task(
    config=tasks_config['quality_assurance'],
    agent=quality_assurance_agent,
    context=[create_content_task], 
    output_pydantic=ContentOutput
)

# Creating the Crew
content_creation_crew = Crew(
    agents=[
        article_fetcher,
        content_creator_agent,
        quality_assurance_agent
    ],
    tasks=[
        fetch_text,
        create_content_task,
        quality_assurance_task
    ],
    verbose=True
)

# Function to perform research with the CrewAI agent
def research_with_crewai(query):
    result = content_creation_crew.kickoff(
        inputs={'url': str(query)},
    )
    # Return the final article result from the structured output.
    return result.pydantic.dict()['article']

# Function to convert text to speech using ElevenLabs
def text_to_speech(text):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 1.0,
            "style": 0.2,
            "use_speaker_boost": True,
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.content  # Returns the audio file bytes
    else:
        st.error("Error converting text to speech: " + response.text)
        return None

def main():
    st.title("Scale Up Your Podcasts with Your Cloned Voice")
    
    # Get user input
    query = st.text_input("Enter your article's URL:")
    
    if st.button("Get Article and Convert to a Podcast"):
        if query:
            # Create a placeholder for running status updates
            status_placeholder = st.empty()
            # Container to store the research result
            result_container = {}

            # Define the worker function that runs CrewAI research
            def run_research():
                result_container['result'] = research_with_crewai(query)
            
            # Start the research in a background thread
            research_thread = threading.Thread(target=run_research)
            research_thread.start()
            
            # Define spinner messages to rotate every 10 seconds
            spinner_messages = [
                "fetching the text of the article",
                "creating podcast narrative",
                "QA of the created podcast narrative", 
                "Finalizing...."
            ]
            msg_index = 0
            # Update the spinner while the research thread is alive
            while research_thread.is_alive():
                status_placeholder.text(spinner_messages[msg_index % len(spinner_messages)])
                time.sleep(10)
                msg_index += 1
            
            # Ensure the thread has completed and clear the status message
            research_thread.join()
            status_placeholder.text("Research complete!")
            
            # Get the final CrewAI result
            crewai_result = result_container.get('result', "No result obtained.")
            
            st.subheader("Podcast Narrative Result:")
            st.write(crewai_result)
            
            with st.spinner("Converting text to speech with your cloned voice..."):
                audio_content = text_to_speech(crewai_result)
            
            if audio_content:
                st.audio(audio_content, format="audio/wav")
        else:
            st.error("Please enter a URL.")

if __name__ == "__main__":
    main()
