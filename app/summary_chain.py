from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import Runnable
from config import OPENAI_API_KEY, MODEL_NAME

def get_summary_chain(style: str = "bullet") -> Runnable:
    """
    Returns a LangChain Runnable that summarizes text using OpenAI based on selected style.
    """
    style_prompt_map = {
        "bullet": "Summarize the following document into {num_points} concise bullet points.",
        "paragraph": "Summarize the following document into {num_points} short paragraphs.",
        "both": "Summarize the following document into {num_points} short paragraph followed by bullet points."
    }

    if style not in style_prompt_map:
        raise ValueError("Invalid summary style. Choose from 'bullet', 'paragraph', or 'both'.")

    prompt = PromptTemplate.from_template(f"""
    {style_prompt_map[style]}
    
    Document:
    {{content}}
    """)

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=MODEL_NAME,
        temperature=0.3
    )

    return prompt | llm
