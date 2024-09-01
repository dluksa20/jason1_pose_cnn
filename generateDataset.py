
import pandas as pd
from utils import IMGSorter

# Define the list of radii
r_list = [10, 15, 20, 30, 40, 50, 60]

# Define the keys for the labels in the dataset
label_keys = ['t1', 't2', 't3', 'r11', 'r21', 'r31', 'r12', 'r22', 'r32']

# List to store the dataset
dataset = []

# Loop through each radius in the list
for r in r_list:

    # Define paths for image and labels based on the radius
    img_path = f'/Users/dovydasluksa/Documents/Project_MSc/Train_img_final/y-axis_FL15_r{r}/renders'
    label_path = f'/Users/dovydasluksa/Documents/Project_MSc/Train_img_final/y-axis_FL15_r{r}/labels'

    # Sort the image and label paths
    sorted_img_paths = IMGSorter(img_path).get_sorted_image_paths(format='_0.png')
    sorted_label_paths = IMGSorter(label_path).get_sorted_image_paths(format='_0.txt')

    # Loop through each pair of image path and label path
    for img_path, label_path in zip(sorted_img_paths, sorted_label_paths):
        # Read label values from the label file
        with open(label_path, 'r') as label_file:
            label_values = [float(val) for val in label_file.read().split()]
            # Create a dictionary with the label keys and values
            label_dict = dict(zip(label_keys, label_values))
            # Append the dictionary with the image path to the dataset
            dataset.append({**label_dict, 'Image path': img_path})
            print(label_path)

# Convert the dataset list into a Pandas DataFrame
df_dataset = pd.DataFrame(dataset)
# Save the DataFrame to a CSV file
df_dataset.to_csv('dataset_LSTM.csv', index=False)

print("Labels saved to dataset.csv")
