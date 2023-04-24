import os
print("11111111111111111", os.getcwd())
# Set the path to the directory containing the files
directory_path = os.getcwd()+'/try_data_1/original_person/'

# Get a list of all the files in the directory
files = os.listdir(directory_path)
files.sort()

# Loop through each file and rename it
for index, file in enumerate(files):
    # Construct the new file name
    # print(index, file)
    new_file_name = f"{index}.jpg"
    # Get the full path of the file
    file_path = os.path.join(directory_path, file)
    # Get the full path of the new file
    new_file_path = os.path.join(directory_path, new_file_name)
    # Rename the file
    os.rename(file_path, new_file_path)
    # print("3")