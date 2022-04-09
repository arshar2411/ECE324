from pathlib import Path 
import json

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def load_data(filename): 
    file_path = str(get_project_root()) + '/data_processing/' + str(filename)
    with open(file_path, "r") as fp:
        data = json.load(fp)

    return data

def save_data(filename, json_dataset): 
    file_path = str(get_project_root()) + '/data_processing/' + str(filename)
    with open(file_path, "w") as fp:
        json.dump(json_dataset, fp, indent=6)
    
    return True 
