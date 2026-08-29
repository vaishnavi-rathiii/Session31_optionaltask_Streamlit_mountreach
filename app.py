import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 Agentic AI Assistant")
st.write("Ask anything and get an AI-powered response.")

# Initialize model
try:
    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2
    )

except Exception as e:
    st.error(f"Model initialization failed: {e}")
    st.stop()

# System prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a very polite and helpful AI assistant. "
        "Give clear and concise answers."
    ),
    ("human", "{user_input}")
])

# User input
user_input = st.text_input(
    "Enter your question:",
    placeholder="What is Artificial Intelligence?"
)

# Generate response
if st.button("Ask AI"):

    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        try:
            formatted_prompt = prompt.invoke({
                "user_input": user_input
            })

            with st.spinner("Thinking..."):
                response = model.invoke(formatted_prompt)

            st.subheader("AI Response")
            st.write(response.content)

        except Exception as e:
            st.error(f"Error: {e}")