from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import Runnable
from config import OPENAI_API_KEY, MODEL_NAME

def get_quiz_chain(difficulty: str = "medium") -> Runnable:
    """
    Returns a LangChain Runnable that generates MCQs from text
    based on selected difficulty level: easy, medium, or hard.
    """
    difficulty_prompt_map = {
        "easy": "Generate {num_questions} basic-level multiple choice questions (MCQs) that test general understanding.",
        "medium": "Generate {num_questions} moderately challenging multiple choice questions (MCQs) that test comprehension and reasoning.",
        "hard": "Generate {num_questions} difficult multiple choice questions (MCQs) that require deep reasoning and inference."
    }

    if difficulty not in difficulty_prompt_map:
        raise ValueError("Invalid difficulty level. Choose from 'easy', 'medium', or 'hard'.")

    prompt = PromptTemplate.from_template(f"""
    {difficulty_prompt_map[difficulty]}

    For each question:
    - Include 4 options labeled A to D
    - Mark the correct answer
    - Provide a brief explanation

    Content:
    {{content}}
    """)

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=MODEL_NAME,
        temperature=0.7  # More creative/flexible for quiz generation
    )

    return prompt | llm
