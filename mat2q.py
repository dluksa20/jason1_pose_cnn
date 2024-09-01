import os
from utils import so3_to_su2, IMGSorter
import numpy as np

# Base directory where the data is stored and will be saved.
BASE_DIR = '/Users/dovydasluksa/Documents/Project_MSc/Train_img_final_4/y-axis_FL5_r{}'

def get_rt_matrix(file_path):
    """
    Reads a file containing the rotation and translation matrix.
    Returns the matrix as a list of lists.
    """
    with open(str(file_path), 'r') as file:
        matrix = [list(map(float, line.split())) for line in file]
    return matrix

def write_to_file(path, data):
    """
    Writes data to a file with each value in the data list being formatted to 6 decimal places.
    """
    with open(path, 'w') as file:
        file.write(" ".join([f"{val:.6f}" for val in data]) + '\n')

if __name__ == '__main__':
    
    # List of radii to process
    r_list = [13]

    for r in r_list:

        # Directories to create
        dir_names = ['rt', 'q', 'pos', 'labels']
        
        # Generate paths based on radii and directory names
        paths = {name: os.path.join(BASE_DIR.format(r), name) for name in dir_names}
        
        # Create directories if they don't exist
        for path in paths.values():
            os.makedirs(path, exist_ok=True)

        # Sort rotation-translation matrix files
        sorted_rt_paths = IMGSorter(paths['rt']).get_sorted_image_paths(format='.txt')

        for path in sorted_rt_paths:
            img_id = os.path.basename(path)

            # Extract rotation and translation matrices from the file
            rtMat = get_rt_matrix(path)
            rt = np.array([rtMat[0][0:3], rtMat[1][0:3], rtMat[2][0:3]])
            
            # Convert rotation matrix to quaternion
            rt_q = so3_to_su2(rt)
            
            # Extract position data
            pos = np.array([rtMat[0][3], rtMat[1][3], rtMat[2][3]])
            
            # Define label based on extracted data
            label = np.array([rtMat[2][3], rtMat[1][3], rtMat[0][3],
                              rtMat[0][0], rtMat[1][0], rtMat[2][0],
                              rtMat[0][1], rtMat[1][1], rtMat[2][1]], dtype=np.float32)
            
            # Write data to files
            write_to_file(os.path.join(paths['pos'], img_id), pos)
            write_to_file(os.path.join(paths['q'], img_id), rt_q)
            write_to_file(os.path.join(paths['labels'], img_id), label)
