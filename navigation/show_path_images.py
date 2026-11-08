import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_path_images(room_path, close_enough, roomloc_filepath, datatrain_dir):

    # Find coordinate of each path
    coord_list = [None] * len(room_path) # Initialise empty list

    with open(roomloc_filepath, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        counter = 0

        for row in reader: # Check each row
            room = row[0]
            loc = row[1]

            roomloc = room + '_' + loc
            if roomloc in room_path: # Check if it is in the list
                x = int(row[2])
                y = int(row[3])

                room_index = room_path.index(roomloc) # Add to coords_list if it is in list
                coord_list[room_index] = np.array([x, y], dtype = int)
                counter += 1

                if counter == len(room_path): # End once all found
                    break

    # Find angle between adjacent roomlocs
    angle_list = []
    for i in range(len(room_path) - 1):

        # Vector to next loc in path
        vec = coord_list[i+1] - coord_list[i]

        # Get angle from 0 to 360
        angle = float(np.atan2(vec[0], vec[1])) * 180 / np.pi
        angle = angle % 360
        angle_list.append(angle)

    # Loop over all data train sets with roomloc to find best image
    best_image = [None] * len(angle_list)
    for room_ind, _ in enumerate(angle_list):

        # Ignore the last one, since we don't need the direction for that
        if room_ind == len(room_path) - 1:
            break

        # Angle we want to find
        angle_true = angle_list[room_ind]

        roomloc = room_path[room_ind]
        pathlist = os.listdir(os.path.join(datatrain_dir, roomloc))
        angle_dist_list = []

        for im_name in pathlist:

            # Extract angle out of filename
            im_split = im_name.split('_')
            im_split = im_split[-1].split('.')
            angle = int(im_split[0])

            # Calculate angular distance between this angle and desired
            diff = abs(angle - angle_true) % 360
            dist = min(diff, 360 - diff)
            angle_dist_list.append(dist)

        # Find the best match
        if min(angle_dist_list) <= close_enough:
            min_index = angle_dist_list.index(min(angle_dist_list))
            best_image[room_ind] = os.path.join(datatrain_dir, roomloc, pathlist[min_index])

    # Display image
    fig, axes = plt.subplots(1, len(best_image), figsize=(12, 4))
    for i, ax in enumerate(axes):
        if best_image[i] is not None:
            img = mpimg.imread(best_image[i])
            ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Towards {room_path[i+1]}')

    plt.suptitle(f'{room_path[0]} to {room_path[-1]}')

if __name__ == '__main__':

    room_path = ['corridor_n209', 
                 'corridor_n233', 
                 'n216_door', 
                 'n216_middle'] # List of roomlocs based on pathfinding
    close_enough = 100 # Angular distance (deg) which is close enough to be a match
    
    roomloc_filepath = 'roomloc_coords.txt'
    datatrain_dir = 'data_train'

    show_path_images(room_path, close_enough, roomloc_filepath, datatrain_dir)

    plt.tight_layout()
    plt.show()