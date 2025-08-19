

import pandas as pd
from test import Retriever
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

def generate_llm_response(query, retrieved_doc, model, tokenizer, chat_history_ids):
    """Generates a response from the LLM."""
    # Prepare the context from the retrieved document
    context = ""
    if retrieved_doc is not None:
        context = f"Here is some context:\n- {retrieved_doc['ParameterDescription']}\n"

    # Create the prompt
    prompt = f"{context}\nUser: {query}"

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
    """Main function to run the conversational search."""
    # Load the data
    file_path = "RAG_rangvalidation/RAG_rangefilter.csv"
    data = pd.read_csv(file_path)

    # Initialize the Retriever
    retriever = Retriever(data)

    # Load the conversational LLM and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    # Start the conversational loop
    print("Welcome to the conversational search!")
    print("You can ask me questions, and I will use the retriever to find relevant information.")

    chat_history_ids = None
    while True:
        # Get user input
        query = input("\nEnter your query (or 'quit' to exit): ")

        # Check if the user wants to quit
        if query.lower() == "quit":
            break

        # Retrieve the best matching document
        retrieved_doc = retriever.retrieve_row(query, data)

        # Get the LLM response
        response, chat_history_ids = generate_llm_response(query, retrieved_doc, model, tokenizer, chat_history_ids)
        print("LLM Response:", response)

        # Ask the user if they want to validate a value
        if retrieved_doc is not None:
            validate_choice = input("Do you want to validate a parameter value? (yes/no): ")
            if validate_choice.lower() == "yes":
                value = input(f"Enter the value for {retrieved_doc['ParameterShortName']} in {retrieved_doc['ManagedObject']}: ")
                allowed_range = retrieved_doc["Range"]
                if validate_input(value, allowed_range):
                    print("\n************************")
                    print("**    Validation Passed!    **")
                    print("************************\n")
                else:
                    print("\n************************")
                    print(f"**   Validation Failed.   **")
                    print(f"** The allowed range is: {allowed_range} **")
                    print("************************\n")

if __name__ == "__main__":
    main()


