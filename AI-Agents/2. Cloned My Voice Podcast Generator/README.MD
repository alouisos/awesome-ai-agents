🚀 I Cloned My Voice To Scale Up My Content With Podcasts (Code)🚀


I'm excited to share a new AI agentic framework that I've been working on—one that harnesses the power of CrewAI and ElevenLabs to turn any article URL into a fully-fledged podcast narrative with your own cloned voice! 🎙️🤖

What It Does:

Dynamic Web Research: The app uses CrewAI to fetch and transform the text of an article into a compelling podcast narrative.

Seamless Text-to-Speech: The narrative is converted into professional-quality audio with my own voice using ElevenLabs' Text-to-Speech API.

User-Friendly Interface: With Streamlit, the entire process is presented in a sleek, interactive web app that updates you in real-time—from fetching text to finalizing your podcast!

Whether you're looking to innovate your content strategy, enhance your podcast production, or simply experiment with AI, this project showcases how modern tools can revolutionize traditional media.

🔗 Check out the video demo and let me know what you think!

https://youtu.be/z65pZ_CLmUQ


Feel free to reach out if you'd like to chat about integrating AI into your media projects!

Below is an example `README.md` file for your GitHub repository:

---

# Scale Up Your Podcasts with AI

This project is a Streamlit web application that transforms an article URL into a compelling podcast narrative with a cloned voice. It leverages the power of **CrewAI** for dynamic web research and **ElevenLabs** for high-quality text-to-speech conversion. All API keys and sensitive credentials are securely loaded from a `.env` file.

## Features

- **Dynamic Web Research:** Uses CrewAI to fetch and process the text of an article.
- **Podcast Narrative Creation:** Transforms the fetched article into an engaging podcast narrative.
- **High-Quality Text-to-Speech:** Converts the narrative into audio using ElevenLabs' TTS API.
- **Real-Time Status Updates:** Displays dynamic progress updates during processing.
- **Secure Credential Management:** Loads all API keys and configurations from a `.env` file.

## Prerequisites

- Python 3.7 or higher
- [Streamlit](https://streamlit.io/)
- [python-dotenv](https://github.com/theskumar/python-dotenv)
- Required Python packages listed in `requirements.txt`

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the Required Packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Create a `.env` File:**  
   In the project root, create a `.env` file and add your API keys and configuration variables:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   SERPER_API_KEY=your_serper_api_key
   OPENAI_MODEL_NAME=gpt-4o-mini
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   ELEVENLABS_VOICE_ID=your_elevenlabs_voice_id
   ```

2. **YAML Configuration Files:**  
   Ensure the `config/agents.yaml` and `config/tasks.yaml` files are correctly configured for your CrewAI agents and tasks.

## Running the Application

To launch the Streamlit app, run:

```bash
streamlit run app.py
```

Once running, enter an article's URL in the input field. The app will then:

1. **Fetch and Process the Article:** Using CrewAI.
2. **Create a Podcast Narrative:** Transforming the article content.
3. **Convert Text-to-Speech:** Generate audio with ElevenLabs.
4. **Display Real-Time Updates:** Spinner messages will indicate each processing stage.

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── config/
│   ├── agents.yaml         # YAML configuration for CrewAI agents
│   └── tasks.yaml          # YAML configuration for CrewAI tasks
├── .env                    # Environment variables for API keys (not committed)
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to modify this `README.md` file to better suit your project's needs. Happy coding!
