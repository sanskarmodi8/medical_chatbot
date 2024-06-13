from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from src.prompt import prompt_template
from store_index import docsearch
import streamlit as st

# Define prompt template and chain type arguments
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","chat_history, "question"])

chain_type_kwargs = {
    "prompt": PROMPT,
    "memory": ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question"
    ),
}

# Load the LLM model (replace with your desired model)
llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML",
                    model_file="llama-2-7b-chat.ggmlv3.q8_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.9})

# Create the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=False,
    chain_type_kwargs=chain_type_kwargs,
)

def main():
    """
    Main function for the Streamlit chat app
    """
    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Center align the page title with reduced spacing
    st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>Medical Chatbot</h1>", unsafe_allow_html=True)

    # Display chat history with visual separation
    chat_container = st.container()
    chat_container.markdown(
        """
        <style>
        .user-message {
            text-align: right;
            background-color: var(--primary-color);
            color: var(--text-color);
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            display: inline-block;
            max-width: 80%;
        }
        .bot-message {
            text-align: left;
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            display: inline-block;
            max-width: 80%;
        }
        @media (prefers-color-scheme: dark) {
            .user-message {
                background-color: #1a73e8;
                color: white;
            }
            .bot-message {
                background-color: #444444;
                color: white;
            }
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Display chat history
    for message in st.session_state.chat_history:
        if message['user'] == 'You':
            chat_container.markdown(
                f"<div class='user-message'><strong>You:</strong> {message['text']}</div>", 
                unsafe_allow_html=True
            )
        else:
            chat_container.markdown(
                f"<div class='bot-message'><strong>Chatbot:</strong> {message['text']}</div>", 
                unsafe_allow_html=True
            )

    # User input field with send button aligned to the right in the same row
    input_col, button_col = st.columns([4, 1])
    with input_col:
        user_input = st.text_input(label="Ask", placeholder="Ask your query...", key="user_input", label_visibility="collapsed")
        
    with button_col:
        # Use on_click callback to handle button click
        if st.button("➤", key="send_button"):
            if user_input:
                # Add user input to chat history
                st.session_state.chat_history.append({"user": "You", "text": user_input})

                # Call the RetrievalQA chain and get response
                response = qa.run({"query": user_input})
                
                # Add response to chat history
                st.session_state.chat_history.append({"user": "Chatbot", "text": response})

                # Rerun the main function to update the UI with new chat history
                st.experimental_rerun()

    # Clear chat history when the page reloads
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()
