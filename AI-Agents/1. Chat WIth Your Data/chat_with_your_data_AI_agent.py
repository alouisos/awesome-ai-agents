import streamlit as st
import pandas as pd
import os
import re
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'wxAgg', etc.
import matplotlib.pyplot as plt
from langchain_experimental.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set the API key from the environment (manage via st.secrets or .env file)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Custom CSS for a refreshed, pleasing color palette
custom_css = """
<style>
body {
    background-color: #f0f4f8;
}
h1 {
    color: #2c3e50;
}
.stButton>button {
    background-color: #007acc;
    color: white;
    border: none;
    padding: 10px 24px;
    font-size: 16px;
    border-radius: 5px;
}
.stButton>button:hover {
    background-color: #005fa3;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

def extract_code_from_response(response):
    """Extracts the python code from the response"""
    code = None
    code_match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if code_match:
        code = code_match.group(1)
    return code

def csv_agent_func(file_path, user_message):
    """Creates an agent that can analyze a csv file and return a response"""
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=OPENAI_API_KEY),
        file_path, 
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )

    try:
        # Format user input
        tool_input = {
            "input": {
                "name": "python",
                "arguments": user_message
            }
        }
        response = agent.run(tool_input)
        return response
    except Exception as e:
        st.write(f"Error: {e}")
        return None

def display_content_from_json(json_response):
    """
    Displays the answer in the Streamlit app based on the JSON response.
    """
    if "answer" in json_response:
        st.write(json_response["answer"])

    if "bar" in json_response:
        data = json_response["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    if "table" in json_response:
        data = json_response["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

def csv_analyzer_app():
    """The streamlit app that allows you to upload a csv file and ask questions about it."""
    st.title('Your Data Analyst AI Agent')
    st.write('Please upload your CSV file and ask me any question:')

    # Initialize query history in session state if not already available
    if "history" not in st.session_state:
        st.session_state.history = []
    
    uploaded_file = st.file_uploader("Select CSV file", type="csv")
    
    if uploaded_file is not None:
        file_details = {
            "File Name": uploaded_file.name,
            "File Type": uploaded_file.type,
            "Size": uploaded_file.size
        }
        st.write(file_details)
        
        # Optional: Save the file to disk
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            df = pd.read_csv(file_path)
            st.dataframe(df)
        except Exception as e:
            st.write(f"Error reading CSV: {e}")
            return
        
        user_input = st.text_input("Ask Your Data A Question")
        if st.button('Execute'):
            with st.spinner("Reasoning..."):
                response = csv_agent_func(file_path, user_input)
            if response is None:
                st.write("No response received.")
                return
            
            # Save the query and response in history
            st.session_state.history.append({"query": user_input, "response": response})
            
            code_to_execute = extract_code_from_response(response)
            
            if code_to_execute:
                try:
                    # Create a local dictionary with df and plt available for execution
                    local_vars = {"df": df, "plt": plt}
                    exec(code_to_execute, globals(), local_vars)
                    fig = plt.gcf()  
                    st.pyplot(fig)  
                except Exception as e:
                    st.write(f"Error during code execution: {e}")
            else:
                st.write(response)
    
    st.divider()
    
    # Display query history
    with st.expander("Query History"):
        if st.session_state.history:
            for idx, entry in enumerate(st.session_state.history):
                st.write(f"**Query {idx+1}:** {entry['query']}")
                st.write(f"**Response:** {entry['response']}")
                st.write("---")
        else:
            st.write("No history available.")

if __name__ == "__main__":
    csv_analyzer_app()
