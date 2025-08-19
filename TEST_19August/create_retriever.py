
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

def load_faiss_index(index_path):
    """Loads a FAISS index from a file."""
    return faiss.read_index(index_path)

def load_data(file_path):
    """Loads the CSV data into a pandas DataFrame."""
    return pd.read_csv(file_path)

def create_retriever_model(model_name='all-MiniLM-L6-v2'):
    """Creates a SentenceTransformer model for retrieval."""
    return SentenceTransformer(model_name)

def retrieve(query, model, index, data, top_k=5):
    """Retrieves the most relevant documents for a given query."""
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, top_k)
    return data.iloc[indices[0]]

def main():
    """Main function to load the retriever and test it."""
    # Load the FAISS index
    index_path = "faiss_index_source.faiss"
    index = load_faiss_index(index_path)

    # Load the data
    file_path = "RAG_rangvalidation/RAG_rangefilter.csv"
    data = load_data(file_path)

    # Create the retriever model
    retriever_model = create_retriever_model()

    # Example usage
    query = "generate rego code for mcc?"
    retrieved_docs = retrieve(query, retriever_model, index, data)

    print("Retrieved documents for the query: '{}'".format(query))
    print(retrieved_docs)

if __name__ == "__main__":
    main()
