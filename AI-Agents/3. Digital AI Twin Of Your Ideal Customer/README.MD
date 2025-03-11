# Chat With Your Ideal Customer Persona

This Streamlit web application generates a company report and an ideal customer persona based on user inputs. It then enables you to chat with a specialized AI agent that “embodies” the generated customer persona. The application leverages CrewAI for content generation and the OpenAI Python API (v1.x) for interactive chat responses.

Video Demo: https://youtu.be/_3yiQYmeF2I

## Features

- **Content Generation:**  
  - Generates a company report including company values and an ideal customer persona using CrewAI agents and tasks.
  
- **Spinner with Dynamic Messages:**  
  - Displays a spinner that cycles through custom messages while waiting for the content generation to complete.

- **Export Functionality:**  
  - Provides a download button to export the generated company report as a TXT file.
  - Provides a download button to export the entire chat history.

- **Interactive Chatbot:**  
  - Uses the OpenAI Python API (v1.x) to enable interactive chat with a persona-specific AI agent.
  - The chatbot context is automatically initialized using the generated customer persona.

## Prerequisites

- **Python 3.7+**  
- **Streamlit**  
- **CrewAI and crewai_tools**  
- **Pydantic**  
- **PyYAML**  
- **OpenAI Python API (v1.x)**  
- **dotenv**  

Ensure you have these libraries installed. For example, you can install them with:

```bash
pip install streamlit crewai crewai-tools pydantic pyyaml openai python-dotenv
```

## Environment Setup

Create a `.env` file in your project root with the following keys:

```
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
OPENAI_MODEL_NAME=your_model_name  # e.g., "gpt-3.5-turbo" or "gpt-4"
```

Also, ensure that you have the required YAML configuration files inside a `config/` folder (e.g., `agents.yaml` and `tasks.yaml`).

## Code Overview

- **Content Generation with CrewAI:**  
  The application creates a CrewAI crew composed of multiple agents and tasks. It processes inputs (company, product, location) to generate two key outputs:
  - **Company Values**  
  - **Customer Persona**

- **Dynamic Spinner:**  
  While the content is being generated, a spinner placeholder updates its message every 10 seconds (configurable) with custom progress messages.

- **Export Buttons:**  
  Once content is generated, download buttons allow you to export:
  - The generated company values and customer persona as a TXT file.
  - The entire chat history as a TXT file.

- **Chatbot Interface:**  
  The application initializes the chat context using the generated customer persona. Users can then interact with the AI agent via a text input. Chat responses are fetched using the OpenAI client (v1.x interface).

## How to Run

To run the application, execute the following command in your terminal:

```bash
streamlit run your_script_name.py
```

Replace `your_script_name.py` with the filename containing your code.

## File Structure

```
.
├── config
│   ├── agents.yaml         # CrewAI agents configuration
│   └── tasks.yaml          # CrewAI tasks configuration
├── .env                    # Environment variables file
├── README.md               # This file
└── app.py                  # Main Streamlit application (contains the code)
```

## Customization

- **Spinner Messages:**  
  Adjust the list of spinner messages and the time interval in the code to suit your needs.

- **Model Selection:**  
  The OpenAI model is specified via the `OPENAI_MODEL_NAME` environment variable. You can change this to switch between models (e.g., from `"gpt-3.5-turbo"` to `"gpt-4"`).

- **Export Options:**  
  The download buttons are implemented using `st.download_button`. You can modify the exported file name or format by editing the corresponding parameters in the code.

## Troubleshooting

- Ensure that your `.env` file is correctly set up with valid API keys.
- Verify that the YAML configuration files are in the correct format and located in the `config/` folder.
- If you encounter issues with the OpenAI API, confirm that you are using version 1.x of the OpenAI Python package and that your API key is valid.

## License

This project is provided under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

This README provides an overview of the project's purpose, how to set up the environment, and detailed instructions on running and customizing the application.