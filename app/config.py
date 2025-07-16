import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Default settings
DEFAULT_SUMMARY_STYLE = "bullet"
DEFAULT_QUIZ_DIFFICULTY = "medium"
MODEL_NAME = "gpt-3.5-turbo"
