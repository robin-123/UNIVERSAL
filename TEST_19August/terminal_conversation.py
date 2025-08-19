
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

def generate_llm_response(query, retrieved_docs, model, tokenizer, chat_history_ids):
    """Generates a response from the LLM."""
    # Prepare the context from retrieved documents
    context = "\n".join([f"- {row['ParameterDescription']}" for _, row in retrieved_docs.iterrows()])

    # Create the prompt
    prompt = f"Here is some context:\n{context}\n\nUser: {query}"

    # Encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    # Create the attention mask
    attention_mask = torch.ones_like(bot_input_ids)

    # Generate a response
    chat_history_ids = model.generate(
        bot_input_ids, 
        max_length=1000, 
        pad_token_id=tokenizer.eos_token_id, 
        attention_mask=attention_mask
    )

    # Decode the response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response, chat_history_ids

def main():
    """Main function to run the conversational RAG system."""
    # Load the FAISS index
    index_path = "faiss_index_source.faiss"
    index = load_faiss_index(index_path)

    # Load the data
    file_path = "RAG_rangvalidation/RAG_rangefilter.csv"
    data = load_data(file_path)

    # Create the retriever model
    retriever_model = create_retriever_model()

    # Load the conversational LLM and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    # Start the conversational loop
    print("Welcome to the conversational RAG system!")
    print("You can ask me questions about the parameters, and I will retrieve relevant information.")

    chat_history_ids = None
    while True:
        # Get user input
        query = input("\nEnter your query (or 'quit' to exit): ")

        # Check if the user wants to quit
        if query.lower() == "quit":
            break
            break

        # Retrieve relevant documents
        retrieved_docs = retrieve(query, retriever_model, index, data)

        # Get the LLM response
        response, chat_history_ids = generate_llm_response(query, retrieved_docs, model, tokenizer, chat_history_ids)
        print("LLM Response:", response)

if __name__ == "__main__":
    main()
