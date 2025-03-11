import streamlit as st
import warnings
warnings.filterwarnings('ignore')

import os
import yaml
import threading
import time
from crewai import Agent, Task, Crew
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import openai
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv()

# Retrieve API keys and model name from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

openai.api_key = OPENAI_API_KEY
# Instantiate an OpenAI client using the new API interface
client = OpenAI(api_key=OPENAI_API_KEY)


# Define structured output schema

class ContentOutputCompany(BaseModel):
    company_description: str = Field(..., description="The company's USP and values, formatted in markdown.")

class ContentOutputMarketResearch(BaseModel):
    market_research: str = Field(..., description="The customer persona, formatted in markdown.")



class ContentOutputPersona(BaseModel):
    customer_persona: str = Field(..., description="The customer persona, formatted in markdown.")




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

search_tool = SerperDevTool(n_results=20)

# Creating Agents
company_content_fetcher = Agent(
    config=agents_config['company_content_fetcher'],
    tools=[search_tool, ScrapeWebsiteTool()],
)

marketing_researcher = Agent(
    config=agents_config['marketing_researcher'],
    tools=[search_tool, ScrapeWebsiteTool()],
)

customer_persona_creator = Agent(
    config=agents_config['customer_persona_creator'],
)

# Creating Tasks
scrape_company_content = Task(
    config=tasks_config['scrape_company_content'],
    agent=company_content_fetcher,
    async_execution=True,
    output_pydantic=ContentOutputCompany
)

marketing_research = Task(
    config=tasks_config['marketing_research'],
    agent=marketing_researcher,
    async_execution=True,
    output_pydantic=ContentOutputMarketResearch
)

create_customer_persona = Task(
    config=tasks_config['create_customer_persona'],
    agent=customer_persona_creator,
    context=[scrape_company_content, marketing_research], 
    output_pydantic=ContentOutputPersona
)

# Creating the Crew
content_creation_crew = Crew(
    agents=[
        company_content_fetcher, 
        marketing_researcher,
        customer_persona_creator,
    ],
    tasks=[
        scrape_company_content, 
        marketing_research,
        create_customer_persona
    ],
    verbose=True
)

def generate_ai_agent(company, product, location):
    # Pass inputs to the CrewAI agent.
    result = content_creation_crew.kickoff(
        inputs={'company': company, 'product': product, 'location': location},
    )
    # Convert the result to a dictionary.
    company_dict = scrape_company_content.output.pydantic.dict()
    market_research_dict = marketing_research.output.pydantic.dict() 
    persona_dict = result.pydantic.dict()

    # Build a dictionary mapping the output keys to their values.
    return {
        "Company": company_dict.get('company_description', ''),
        "Market Research": market_research_dict.get('market_research', ''),
        "Customer Persona": persona_dict.get('customer_persona', ''),
    }


def get_chatgpt_response(messages):
    """
    Sends messages to ChatGPT using the new client interface and returns the assistant's reply.
    """
    response = client.chat.completions.create(
       model=OPENAI_MODEL_NAME,
       messages=messages,
       temperature=0.7,
    )
    # Access the content via .message.content on the pydantic model
    return response.choices[0].message.content

# Streamlit App Layout
st.title("Chat With Your Ideal Customer Persona")
st.write("Enter the name of your company, your product description, and the location that your persona lives to generate a company report and a description of your ideal customer persona. Then chat with a specialized AI agent that talks and feels like  your customer personna. Bounce marketing ideas, get opinions, feelings, nemeses etc")

company_input = st.text_input("Company Nme")
product_input = st.text_input("Product Description")
location_input = st.text_input("Location Your Customer Lives e.g. New York, United States or just United States")


if st.button("Generate"):
    if product_input and location_input:
        # Create a placeholder for the spinner message.
        spinner_placeholder = st.empty()
        # Define the list of messages to cycle through.
        spinner_messages = [
            "Searching for the company...",
            "Analyzing company USP and values...",
            "Searching for the ideal customer profile for the company...",
            "Creating ideal customer persona...", 
            "Finalizing output...", 
            "Finalizing output...", 
            "Finalizing output...", 
            "Finalizing output..."
        ]
        result_container = {}

        # Function to run the generation in a separate thread.
        def run_generation():
            result_container["output"] = generate_ai_agent(company_input, product_input, location_input)

        # Start the generation in a new thread.
        thread = threading.Thread(target=run_generation)
        thread.start()

        # Update the spinner message every 2 seconds until the thread finishes.
        message_index = 0
        while thread.is_alive():
            spinner_placeholder.text(spinner_messages[message_index % len(spinner_messages)])
            message_index += 1
            time.sleep(10)  # Adjust the interval (in seconds) as needed.
        thread.join()
        spinner_placeholder.empty()  # Clear the spinner

        output = result_container.get("output", {})
        st.markdown("## Company Values")
        st.markdown(output.get("Company", ""))
        st.markdown("## Market Research")
        st.markdown(output.get("Market Research", ""))
        st.markdown("## Customer Persona")
        st.markdown(output.get("Customer Persona", ""))


        company_output = output.get("Company", "")
        market_research_output = output.get("Market Research", "")
        customer_output = output.get("Customer Persona", "")

        # Prepare the text content to be exported
        export_text = f"Company Values:\n{company_output}\n\nMarket Research:\n{market_research_output}\n\nCustomer Persona:\n{customer_output}"

        # Add a download button to export the outputs as a txt file
        st.download_button(
            label="Download as TXT",
            data=export_text,
            file_name="output.txt",
            mime="text/plain"
        )

        # Initialize chat context using the generated output
        context_prompt = f"""The following is the generated output from our analysis:

Customer Persona:
{output.get("Customer Persona", "")}

Use this information as context to help answer questions and discuss company branding and customer insights.
Pretend that you are this person as described in the customer persona. 
Feel like the customer persona, talk like the customer persona, respond like the customer persona. 
"""
        # Initialize or reset the chat history in session state.
        st.session_state.chat_history = [{"role": "system", "content": context_prompt}]
        st.success("Your Ideal Customer Persona Chatbot has been created! Start chatting below!")
    else:
        st.error("Please enter both product and location.")

# Chatbot interface
if "chat_history" in st.session_state:
    st.subheader("Chat with your ideal customer persona")
    
    # Display previous conversation
    for message in st.session_state.chat_history:
        if message["role"] != "system":
            st.markdown(f"**{message['role'].capitalize()}:** {message['content']}")

    # Add a download button for chat history
    # Convert chat history list into a string with each message on a new line
    chat_history_text = "\n\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history]
    )
    st.download_button(
        label="Download Chat History",
        data=chat_history_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )
    
    user_message = st.text_input("Your message", key="user_message")
    if st.button("Send") and user_message:
        # Append user's message to conversation history
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        
        with st.spinner("Thinking..."):
            # Get assistant's response using the conversation history
            assistant_reply = get_chatgpt_response(st.session_state.chat_history)
        
        # Append assistant's response to conversation history and display it
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
        st.markdown(f"**Persona:** {assistant_reply}")