import pandas as pd
import difflib

def validate_input(value, allowed_range):
    """
    Validates the user input against an allowed range dynamically.
    Returns a tuple (is_valid, suggestion).
    """
    if pd.isna(allowed_range):
        return (False, "No validation rule defined.")

    if not isinstance(allowed_range, str):
        allowed_range = str(allowed_range)

    allowed_range = allowed_range.strip()

    # Case 1: "X chars" for string length validation
    if "chars" in allowed_range:
        try:
            length = int(allowed_range.split()[0])
            if len(str(value)) == length:
                return (True, None)
            else:
                return (False, f"The value must have {length} characters.")
        except (ValueError, IndexError):
            return (False, "Invalid range format for character length.")

    # Case 2: "min..max" for numerical range validation
    elif ".." in allowed_range:
        try:
            parts = allowed_range.split("..", 1)
            min_val = int(parts[0])
            max_val_part = parts[1].split(" ")[0]
            max_val = int(max_val_part)
            if min_val <= int(value) <= max_val:
                return (True, None)
            else:
                return (False, f"Please enter a value between {min_val} and {max_val}.")
        except (ValueError, TypeError, IndexError):
            return (False, "Invalid range format for numerical range.")
            
    # Case 3: Comma-separated list of allowed values
    elif "," in allowed_range:
        allowed_values = [item.strip() for item in allowed_range.split(',')]
        if str(value) in allowed_values:
            return (True, None)
        else:
            return (False, f"Please choose one of the following values: {', '.join(allowed_values)}")

    # Case 4: A single number for exact match
    else:
        try:
            expected_value = int(allowed_range)
            if int(value) == expected_value:
                return (True, None)
            else:
                return (False, f"The only allowed value is {expected_value}.")
        except (ValueError, TypeError):
            # If not a number, treat as a string for exact match
            if str(value) == allowed_range:
                return (True, None)
            else:
                return (False, f"The only allowed value is '{allowed_range}'.")

def main():
    """
    Main function to run the MO type validation.
    """
    # Load the data
    file_path = "RAG_rangvalidation/RAG_rangefilter.csv"
    try:
        data = pd.read_csv(file_path, encoding='latin1')
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    unique_mo_types = data['ManagedObject'].unique().tolist()

    # Start the conversational loop
    while True:
        mo_choice = input("\nEnter the Managed Object Type (or 'quit' to exit): ")
        if mo_choice.lower() == 'quit':
            break

        if mo_choice in unique_mo_types:
            selected_mo_type = mo_choice
            mo_filtered_data = data[data['ManagedObject'] == selected_mo_type]
            unique_parameters = mo_filtered_data['ParameterShortName'].unique().tolist()

            while True:
                param_choice = input(f"Enter the Parameter Short Name for {selected_mo_type} (or 'back' to select MO type again): ")
                if param_choice.lower() == 'back':
                    break

                if param_choice in unique_parameters:
                    selected_parameter = param_choice
                    
                    # Get the specific row for validation
                    validation_row = mo_filtered_data[mo_filtered_data['ParameterShortName'] == selected_parameter].iloc[0]
                    
                    value = input(f"Enter the value for {selected_parameter} in {selected_mo_type}: ")
                    allowed_range = validation_row["Range"]
                    
                    is_valid, suggestion = validate_input(value, allowed_range)
                    if is_valid:
                        print("\n************************")
                        print("**    Validation Passed!    **")
                        print("************************")
                    else:
                        print("\n************************")
                        print(f"**   Validation Failed.   **")
                        if suggestion:
                            print(f"** Suggestion: {suggestion} **")
                        else:
                            print(f"** The allowed range is: {allowed_range} **")
                        print("************************")
                    # After validation, go back to parameter selection
                    continue
                else:
                    suggestions = difflib.get_close_matches(param_choice, unique_parameters)
                    print(f"Parameter '{param_choice}' not found for MO Type '{selected_mo_type}'.")
                    if suggestions:
                        print("Did you mean one of these?")
                        for s in suggestions:
                            print(f"- {s}")
                    else:
                        print("Available parameters are:")
                        for p in unique_parameters:
                            print(f"- {p}")

        else:
            suggestions = difflib.get_close_matches(mo_choice, unique_mo_types)
            print(f"MO Type '{mo_choice}' not found.")
            if suggestions:
                print("Did you mean one of these?")
                for s in suggestions:
                    print(f"- {s}")

if __name__ == "__main__":
    main()
