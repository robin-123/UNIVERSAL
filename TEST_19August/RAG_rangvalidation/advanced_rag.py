import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_data(file_path):
    """Loads the CSV data into a pandas DataFrame."""
    return pd.read_csv(file_path)

def create_embeddings(data, model_name='all-MiniLM-L6-v2'):
    """Creates embeddings for the 'ParameterDescription' column."""
    model = SentenceTransformer(model_name)
    descriptions = data['ParameterDescription'].fillna('').tolist()
    embeddings = model.encode(descriptions, convert_to_tensor=True)
    return embeddings.cpu().numpy()

def build_faiss_index(embeddings):
    """Builds a FAISS index from the embeddings."""
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query, model, index, data, top_k=5):
    """Retrieves the most relevant documents for a given query."""
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, top_k)
    return data.iloc[indices[0]]

def validate_input(value, allowed_range):
    """Validates the user input against the allowed range."""
    if isinstance(allowed_range, str):
        if allowed_range == "3 chars":
            return len(str(value)) == 3
        elif "-" in allowed_range:
            try:
                min_val, max_val = map(int, allowed_range.split("-"))
                return min_val <= int(value) <= max_val
            except ValueError:
                return False
    return False

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
    """Main function to run the advanced RAG system."""
    # Load the data
    file_path = "RAG_rangefilter.csv"
    data = load_data(file_path)

    # Create embeddings and build the FAISS index
    embeddings = create_embeddings(data)
    index = build_faiss_index(embeddings)

    # Create the model for encoding queries
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load the conversational LLM and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    # Start the conversational loop
    print("Welcome to the advanced RAG system with a real LLM!")
    print("You can ask me questions about the parameters, and I will retrieve relevant information.")
    print("You can also ask me to validate a parameter value.")

    chat_history_ids = None
    while True:
        # Get user input
        query = input("\nEnter your query (or 'quit' to exit): ")

        # Check if the user wants to quit
        if query.lower() == "quit":
            break

        # Retrieve relevant documents
        retrieved_docs = retrieve(query, retriever_model, index, data)

        # Get the LLM response
        response, chat_history_ids = generate_llm_response(query, retrieved_docs, model, tokenizer, chat_history_ids)
        print("LLM Response:", response)

        # Ask the user if they want to validate a value
        validate_choice = input("Do you want to validate a parameter value? (yes/no): ")

        if validate_choice.lower() == "yes":
            managed_object = input("Enter the ManagedObject: ")
            parameter = input("Enter the ParameterShortName: ")
            value = input(f"Enter the value for {parameter} in {managed_object}: ")

            try:
                # Get the allowed range for the parameter
                allowed_range = data[(data["ManagedObject"] == managed_object) & (data["ParameterShortName"] == parameter)]["Range"].iloc[0]

                # Validate the input
                if validate_input(value, allowed_range):
                    print("\n************************")
                    print("**    Valid input!    **")
                    print("************************\n")
                else:
                    print("\n************************")
                    print(f"**   Invalid input.   **")
                    print(f"** The allowed range is: {allowed_range} **")
                    print("************************\n")
            except IndexError:
                print("Invalid ManagedObject or ParameterShortName.")

if __name__ == "__main__":
    main()