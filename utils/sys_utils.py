import os
from dotenv import load_dotenv


def print_environment_variables():
    for key, value in os.environ.items():
        print(f"{key}: {value}")

print_environment_variables()

def get_files_by_extension(directory, extension):
    file_paths = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    
    return file_paths

def calculate_directory_size(directory):
    total_size = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    
    # Convert size to GB
    total_size_gb = total_size / (1024**3)
    
    return total_size_gb


