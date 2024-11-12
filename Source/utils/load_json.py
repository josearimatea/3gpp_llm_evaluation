import json

def load_responses(file_path):
    """
    Load responses from a JSON file.

    :param file_path: The path to the JSON file.
    :return: The data loaded from the JSON file.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_path}.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
