import pandas as pd

def load_data(file_path):
    """Loads the CSV data into a pandas DataFrame."""
    return pd.read_csv(file_path)

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

def main():
    """Main function to run the RAG conversation."""
    # Load the data
    file_path = "RAG_rangefilter.csv"
    data = load_data(file_path)

    # Get the list of managed objects
    managed_objects = data["ManagedObject"].unique()

    # Start the conversation
    print("Welcome to the RAG conversation!")
    print("I can help you validate parameter values.")

    while True:
        # Get user input for ManagedObject
        print("Available Managed Objects:")
        for mo in managed_objects:
            print(f"- {mo}")
        managed_object = input("Enter the Managed Object (or 'quit' to exit): ")

        # Check if the user wants to quit
        if managed_object.lower() == "quit":
            break

        # Check if the ManagedObject is valid
        if managed_object not in managed_objects:
            print("Invalid Managed Object. Please choose from the list above.")
            continue

        # Get the list of parameters for the selected ManagedObject
        parameters = data[data["ManagedObject"] == managed_object]["ParameterShortName"].unique()
        print(f"Available parameters for {managed_object}:")
        for param in parameters:
            print(f"- {param}")

        # Get user input for ParameterShortName
        parameter = input(f"Enter the parameter name for {managed_object} (or 'back' to choose another Managed Object): ")

        # Check if the user wants to go back
        if parameter.lower() == "back":
            continue

        # Check if the parameter is valid
        if parameter not in parameters:
            print("Invalid parameter. Please choose from the list above.")
            continue

        # Get the allowed range for the parameter
        allowed_range = data[(data["ManagedObject"] == managed_object) & (data["ParameterShortName"] == parameter)]["Range"].iloc[0]

        # Get the user value
        value = input(f"Enter the value for {parameter} in {managed_object}: ")

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

if __name__ == "__main__":
    main()