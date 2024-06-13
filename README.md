# Medical Chatbot using Llama2

## Overview

This project implements a medical chatbot using Streamlit for the user interface and LangChain for natural language processing (NLP) capabilities. The chatbot aims to provide information and answer questions related to various medical conditions and treatments based on a predefined knowledge base.

## Features

- **Interactive Chat Interface**: Users can interact with the chatbot in a conversational manner.
- **Knowledge Base Integration**: Utilizes LangChain for retrieving information from a structured knowledge base.
- **Persistent Chat History**: Maintains a session-based chat history for a seamless user experience across interactions.

## Technologies Used

- **Streamlit**: Web application framework used for building the chat interface.
- **LangChain**: Library for handling natural language processing tasks, including retrieval-based question answering.
- **Python Libraries**: Various Python libraries such as Flask, Pinecone Client, and others for backend processing and integration.
- **Pinecone**: Used for vector storage and retrieval in the knowledge base setup.

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/sanskarmodi8/medical_chatbot.git
   cd medical_chatbot
   ```

2. **Install Dependencies**

   Ensure you have Python 3.8+ and create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Run the Application**

   ```bash
   streamlit run streamlit_app.py
   ```

   This will start the Streamlit server. Open your browser and go to `http://localhost:8501` to interact with the chatbot.

## Usage

- Enter your query or question in the input field and click the ➤ button or press Enter to send.
- The chatbot will process the input in some time using LangChain and retrieve relevant information or respond accordingly.
- Chat history is displayed in the interface, showing both user queries and chatbot responses.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository and create your branch from `main`.
2. Make your changes and test thoroughly.
3. Submit a pull request detailing your changes and any relevant information.

## Contact

For questions or support, feel free to contact [Sanskar Modi](mailto:sansyprog8@gmail.com).
