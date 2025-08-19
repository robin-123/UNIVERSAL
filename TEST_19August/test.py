import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

class Retriever:
    def __init__(self, df):
        """
        Initializes the Retriever class with a DataFrame.
        
        :param df: DataFrame containing the dataset
        """
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.df = df

    def load_faiss_index(self, index_file):
        """
        Loads the FAISS index for the given usecase from the specified directory.

        :param usecase: The name of the usecase (corresponding to the FAISS index file)
        :return: Loaded FAISS index
        """
        try:
            # Load the FAISS index from file
            index = faiss.read_index(index_file)
            print(f"FAISS index loaded from {index_file}")
            return index
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return None

    def retrieve_row(self, query, grouped_df, k=3):
        """
        Retrieves the top-k closest documents from the FAISS index based on the user query.
        
        :param query: The user query (string)
        :param k: Number of closest matches to retrieve (default is 3)
        :return: Context dictionary for the top match
        """
        # Load the appropriate FAISS index for the usecase
        index = self.load_faiss_index("faiss_index_source.faiss")
        if index is None:
            return None
        
        # Encode the user query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
 
	# Ensure correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

	# Check for dimension match
        if query_embedding.shape[1] != index.d:
                print(f"Dimension mismatch: query {query_embedding.shape[1]} vs index {index.d}")
                return None

	# Search the FAISS index
        D, I = index.search(query_embedding, k)

        print("Distances:", D)
        print("Indices:", I)

	# Filter out -1s
        valid_indices = I[0][I[0] != -1]
        if len(valid_indices) == 0:
                print("No valid matches found.")
                return None

        matched_rows = grouped_df.iloc[valid_indices]	
        best_match_row = matched_rows.iloc[0]


        print("Matched rows:", matched_rows)
        return matched_rows

        # Create the context for the response template (this is based on the best match)
        #context = self.create_context(best_match_row)
        
        #return context

    def retreive_response_template(self, query, k=3):
	      # Load the appropriate FAISS index for the usecase
        index = self.load_faiss_index("faiss_index_dataset.faiss")
        if index is None:
            return None
        
        # Encode the user query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)

        # Search the FAISS index for the top-k closest documents
        D, I = index.search(query_embedding, k)
        results = [responses[i] for i in I[0]]
        return results[0] if results else "No matching response found."

    def operator_mapping(self, raw_op):
        op_mapping = {
            "EQUALS": "==",
            "NOT_EQUALS": "!=",
            "GREATER_THAN": ">",
            "LESS_THAN": "<",
            "GREATER_THAN_OR_EQUALS": ">=",
            "LESS_THAN_OR_EQUALS": "<=",
        }
        operation_symbol = op_mapping.get(raw_op, raw_op)
        return operation_symbol

    def fetch_default_val(self,row):
        #raw_operator = row["Operation"].strip().upper()
        #operation_symbol = self.operator_mapping(raw_operator)
        return row["DefaultValue"]

    def create_context(self, row, raw_op, val):
        """
        Creates a context dictionary from the best match row.
        
        :param row: The best match row from the DataFrame
        :return: Context dictionary
        """
        #raw_operator = row["Operation"].strip().upper()
        operation_symbol = self.operator_mapping(raw_op)

        if pd.isna(row["DistinguishName"]):
            attribute = f"{row['ManagedObject']}.{row['ParameterShortName']}"
        else:
            DistinguishName = row['DistinguishName'].replace("/",".")
            attribute = f"{DistinguishName}.{row['ParameterShortName']}"

        # Context for the response
        context = {
            "Vendor": row["Vendor"],
            "MOType": row["ManagedObject"],
            "Attribute": attribute,
            "Value": val,
            "Operation": operation_symbol
        }
        
        return context

    def get_response_template(self, usecase):
        """
        Fetch the response template for a given usecase.
        
        :param usecase: The name of the usecase for which the template is required
        :return: The response template (string)
        """
        try:
            # Load response templates from a JSON file or mapping
            response_templates = self.load_response_templates()

            # Fetch the template for the given usecase
            response_template = response_templates.get(usecase, "No template found for this usecase")
            return response_template
        except Exception as e:
            print(f"Error fetching response template for {usecase}: {e}")
            return "Error: Template not found"

    def load_response_templates(self):
        """
        Loads response templates from a JSON file or pre-defined mapping.
        
        :return: Dictionary with usecase -> response_template mapping
        """
        try:
            # Example: Load templates from a JSON file (assuming a mapping of usecases to templates)
            with open("response_templates.json", "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading response templates: {e}")
            return {}

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

if __name__ == "__main__":
    # Load the data
    file_path = "RAG_rangvalidation/RAG_rangefilter.csv"
    data = pd.read_csv(file_path)

    # Initialize the Retriever (not strictly needed for this flow, but keeping for consistency)
    retriever = Retriever(data)

    # Start the conversational loop
    while True:
        # Directly ask for MO type
        unique_mo_types = data['ManagedObject'].unique().tolist() # Get all unique MO types from the entire data
        print("\nAvailable Managed Object Types:")
        for i, mo_type in enumerate(unique_mo_types):
            print(f"{i + 1}. {mo_type}")

        mo_choice = input("Select a Managed Object Type by number (or 'quit' to exit): ")
        if mo_choice.lower() == 'quit':
            break

        try:
            mo_index = int(mo_choice) - 1
            if 0 <= mo_index < len(unique_mo_types):
                selected_mo_type = unique_mo_types[mo_index]
                
                # Filter data for the selected MO type
                mo_filtered_data = data[data['ManagedObject'] == selected_mo_type]
                unique_parameters = mo_filtered_data['ParameterShortName'].unique().tolist()

                print(f"\nParameters for {selected_mo_type}:")
                for i, param in enumerate(unique_parameters):
                    print(f"{i + 1}. {param}")

                while True:
                    try:
                        param_choice = input("Select a Parameter by number (or 'back' to select MO type again): ")
                        if param_choice.lower() == 'back':
                            break
                        param_index = int(param_choice) - 1
                        if 0 <= param_index < len(unique_parameters):
                            selected_parameter = unique_parameters[param_index]
                            
                            # Get the specific row for validation
                            validation_row = mo_filtered_data[mo_filtered_data['ParameterShortName'] == selected_parameter].iloc[0]
                            
                            value = input(f"Enter the value for {selected_parameter} in {selected_mo_type}: ")
                            allowed_range = validation_row["Range"]
                            
                            if validate_input(value, allowed_range):
                                print("\n************************")
                                print("**    Validation Passed!    **")
                                print("************************")
                            else:
                                print("\n************************")
                                print(f"**   Validation Failed.   **")
                                print(f"** The allowed range is: {allowed_range} **")
                                print("************************")
                            # After validation, go back to parameter selection
                            continue 
                        else:
                            print("Invalid parameter number. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter a number or 'back'.")
            else:
                print("Invalid MO type number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'quit'.")