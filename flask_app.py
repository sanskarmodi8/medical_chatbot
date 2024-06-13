from flask import Flask, render_template, request
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from src.prompt import prompt_template
from store_index import docsearch

app = Flask(__name__)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","chat_history, "question"])

chain_type_kwargs={
        "prompt": PROMPT,
        "memory": ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question"),
    }

llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML",
                    model_file="llama-2-7b-chat.ggmlv3.q8_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.9})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=False,
    chain_type_kwargs=chain_type_kwargs,
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    inp = msg
    print(inp)
    result = qa.invoke({"query":inp})['result']
    print("Response : ", result)
    return str(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
    
    

