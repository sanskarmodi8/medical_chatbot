from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
import pickle
import uuid
from src.custom_pinecone import CustomPinecone

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

if os.path.getsize("extracted_data.pkl")==0:
    extracted_data = load_pdf("../data/Gale Encyclopedia of Medicine All 5 Volumes Combined.pdf")
    with open("extracted_data.pkl", "wb") as f:
        pickle.dump(extracted_data, f)
else:
    with open("extracted_data.pkl", "rb") as f:
        extracted_data = pickle.load(f)
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)



def langchain_pinecone_from_texts_custom_updated(
    texts,
    embedding,
    metadatas= None,
    ids= None,
    batch_size= 32,
    text_key="text",
    index_name= None,
    namespace= None,
    
) -> Pinecone:
    
    vector_count=0

    indexes = pc.list_indexes().names()  # checks if provided index exists

    if index_name in indexes:
        index = pc.Index(index_name)
        index_stats = index.describe_index_stats()
        vector_count = index_stats['total_vector_count']
        print(vector_count)
    elif len(indexes) == 0:
        raise ValueError(
            "No active indexes found in your Pinecone project, "
            "are you sure you're using the right API key and environment?"
        )
    else:
        raise ValueError(
            f"Index '{index_name}' not found in your Pinecone project. "
            f"Did you mean one of the following indexes: {', '.join(indexes)}"
        )

    if vector_count<44000 :
        for i in range(0, len(texts), batch_size):
            # set end position of batch
            i_end = min(i + batch_size, len(texts))
            # get batch of texts and ids
            lines_batch = texts[i:i_end]
            # create ids if not provided
            if ids:
                ids_batch = ids[i:i_end]
            else:
                ids_batch = [str(uuid.uuid4()) for n in range(i, i_end)]
            # create embeddings
            embeds = embedding.embed_documents(lines_batch)
            # prep metadata and upsert batch
            if metadatas:
                metadata = metadatas[i:i_end]
            else:
                metadata = [{} for _ in range(i, i_end)]
            for j, line in enumerate(lines_batch):
                metadata[j][text_key] = line
            to_upsert = zip(ids_batch, embeds, metadata)

            # upsert to Pinecone
            index.upsert(vectors=list(to_upsert), namespace=namespace)
    return Pinecone(index, embedding.embed_query, text_key, namespace)
        
docsearch = langchain_pinecone_from_texts_custom_updated([t.page_content for t in text_chunks], embeddings, index_name="medical-chatbot")
# docsearch.__class__ = CustomPinecone
