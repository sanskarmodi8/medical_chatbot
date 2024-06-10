from langchain.vectorstores import Pinecone
from langchain.docstore.document import Document

class CustomPinecone(Pinecone):
    def similarity_search_with_score(self, query, k, filter=None, namespace=None):
        # updated query call
        query_obj = self._embedding_function(query)
        docs=[]
        results = self._index.query(
            vector=[query_obj],
            top_k=k,
            include_metadata=True,
            namespace=namespace,
            filter=filter,
        )
        for res in results["matches"]:
            metadata = res["metadata"]
            if self._text_key in metadata:
                text = metadata.pop(self._text_key)
                score = res["score"]
                docs.append((Document(page_content=text, metadata=metadata), score))
            
        return docs