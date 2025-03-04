# Your Data Analyst AI Agent

Welcome to **Your Data Analyst AI Agent** – your very own intelligent data analyst that can analyze CSV files, generate visualizations, and answer your data questions on the fly! This exciting Streamlit application leverages the power of OpenAI and LangChain to bring cutting-edge AI directly to your desktop.

## Features

- **Interactive CSV Upload:** Easily upload your CSV files and explore your data.
- **Smart Query Response:** Ask questions about your data and receive intelligent responses powered by AI.
- **Dynamic Visualizations:** Automatically generate charts and tables based on your queries.
- **Query History:** Keep track of all your interactions for later review.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.7 or higher
- Pip (Python package installer)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/data-analyst-ai-agent.git
   cd data-analyst-ai-agent
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   Make sure you have a `requirements.txt` file in the repository (see the provided file for dependencies). Install them with:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**

   Create a `.env` file in the root directory of your project and add your OpenAI API key:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## How to Run

Start the Streamlit app with the following command:

```bash
streamlit chat_with_your_data_AI_agent.py
```

Once the app is running, open the provided URL (usually `http://localhost:8501`) in your browser. Upload your CSV file, type in your data query, and let the AI work its magic!

## How It Works

1. **CSV Upload & Display:** The application allows you to upload a CSV file, which it then reads and displays.
2. **Data Query Processing:** When you input a question about your data, the code sends your query to an AI agent powered by LangChain and OpenAI.
3. **Response Handling:** The AI agent processes your question, generates a response, and if the response includes Python code, it executes it to generate visualizations.
4. **Visualization & History:** Results are displayed in an interactive format (charts, tables) along with a history of your queries.

Check the video demo here: 
https://youtu.be/FhRUhOVOL2A

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request if you have improvements or new ideas.

## License

This project is licensed under the [MIT License](LICENSE).

---

Embrace the future of data analysis – your data analyst is now just a click away. Enjoy exploring and gaining insights from your data like never before!
