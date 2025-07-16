import streamlit as st
from loaders import load_document
from summary_chain import get_summary_chain
from quiz_chain import get_quiz_chain
from config import DEFAULT_SUMMARY_STYLE, DEFAULT_QUIZ_DIFFICULTY
from langchain_core.runnables import RunnableParallel

st.set_page_config(page_title="DocIntel", layout="centered")

st.title("📄 DocIntel")
st.subheader("Unlock insights from your documents — summarize and quiz with AI.")

# File uploader
uploaded_file = st.file_uploader("Upload your document (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

# Task selection
task = st.radio("What would you like to do?", ["Summarize", "Generate Quiz", "Both"], horizontal=True)

# Task-specific options
with st.expander("Customize Options"):
    if task in ["Summarize", "Both"]:
        summary_style = st.selectbox("Summary Style", ["bullet", "paragraph", "both"], index=0)
        summary_length = st.slider("Number of summary points/paragraphs", min_value=1, max_value=15, value=7)
    if task in ["Generate Quiz", "Both"]:
        quiz_difficulty = st.selectbox("Quiz Difficulty", ["easy", "medium", "hard"], index=1)
        num_questions = st.slider("Number of quiz questions", min_value=1, max_value=15, value=5)


# Generate button
if uploaded_file and st.button("Generate"):
    with st.spinner("Processing your document..."):
        try:
            # Load and prepare text
            docs = load_document(uploaded_file)
            full_text = "\n".join([doc.page_content for doc in docs])
            
            # Prepare input for chains
            input_data = {"content": full_text}

            if task in ["Summarize", "Both"]:
                input_data["num_points"] = summary_length
            if task in ["Generate Quiz", "Both"]:
                input_data["num_questions"] = num_questions

            # Build chains
            chains = {}
            if task in ["Summarize", "Both"]:
                chains["summary"] = get_summary_chain(style=summary_style)
            if task in ["Generate Quiz", "Both"]:
                chains["quiz"] = get_quiz_chain(difficulty=quiz_difficulty)

            # Run chains
            if len(chains) == 1:
                result_key = list(chains.keys())[0]
                result = {result_key: chains[result_key].invoke(input_data)}
            else:
                parallel_chain = RunnableParallel(chains)
                result = parallel_chain.invoke(input_data)

            # Output
            def render_output(label, content):
                st.markdown(f"### {label}")
                text = content.content if hasattr(content, "content") else str(content)
                st.markdown(
                    f"<div style='color: #F0F0F0; font-size: 16px; line-height: 1.6; white-space: pre-wrap;'>{text}</div>",
                    unsafe_allow_html=True
                )

            if "summary" in result:
                render_output("Summary", result["summary"])
            if "quiz" in result:
                render_output("Quiz", result["quiz"])

        except Exception as e:
            st.error(f" An error occurred: {e}")
