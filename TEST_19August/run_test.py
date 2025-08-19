
import pandas as pd
from test import Retriever

def main():
    """Main function to test the Retriever class."""
    # Create a dummy DataFrame for initialization
    dummy_df = pd.DataFrame()

    # Initialize the Retriever
    retriever = Retriever(dummy_df)

    # Load the FAISS index
    index_file = "faiss_index_source.faiss"
    index = retriever.load_faiss_index(index_file)

    if index:
        print("FAISS index loaded successfully.")

if __name__ == "__main__":
    main()
