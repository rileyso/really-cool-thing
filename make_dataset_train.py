import numpy as np
import cv2
import os
import csv

def make_dataset_train(settings_filepath, 
                        vid_filepath, training_folder_dir, 
                        room, loc, 
                        initial_direction, end_direction, 
                        cut_start, cut_end, 
                        n_directions):

    # Check to see if the video already exists in the settings file
    existing = False
    with open(settings_filepath, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == vid_filepath: # Find a matching video file path in the settings
                print('Found. Reading from settings file.')
                room = row[1]
                loc = row[2]
                initial_direction = int(row[3])
                end_direction = int(row[4])
                cut_start = float(row[5])
                cut_end = float(row[6])
                n_directions = int(row[7])

                existing = True
                break

    if not existing:
        print('No corresponding settings found. Will save as new data.')

    # Confirm settings used:
    print(f"Using settings: {vid_filepath,room,loc,initial_direction,end_direction,cut_start,cut_end,n_directions}")

    # Where to save the training data
    output_dir = os.path.join(training_folder_dir,  room + '_' + loc)

    # Make path if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    # Make directions
    frame_directions = np.round((np.linspace(initial_direction, end_direction, n_directions))).astype(int)

    # Extract frames
    vid = cv2.VideoCapture(vid_filepath)
    if not vid.isOpened():
        raise Exception("Error: Could not open video {vid_filepath}")

    # Get video characteristics
    fps = vid.get(cv2.CAP_PROP_FPS)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        
    # Start and ending frame indices
    start_frame = int(cut_start * fps)
    end_frame = int(cut_end * fps)
    if cut_end == -1:
        end_frame = total_frames - 1

    if start_frame >= total_frames:
        vid.release()
        raise Exception("Error: start_frame ({start_frame}) is greater than total_frames ({total_frames})")
    if end_frame > total_frames:
        vid.release()
        raise Exception("Error: end_frame ({cut_end}) is greater than total_frames ({total_frames})")

    # Make linearly spaced frame indices
    frame_indices = [int(start_frame + i * (end_frame - start_frame) / (n_directions - 1)) for i in range(n_directions)]

    # Extract the frames out and save them
    for i, frame_index in enumerate(frame_indices):

        # Go to frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read frame
        success, frame = vid.read()
        if not success:
            raise Exception(f"Error: Could not read frame at index {frame_index}")

        # Rotate frame if it is in the wrong direction
        height, width = frame.shape[:2]
        if width > height:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Get current direction
        current_dir = frame_directions[i]

        # Save frame
        output_filename = os.path.join(output_dir, room + '_' + loc + '_im' + str(i) + '_' + str(current_dir) + '.jpg')
        print(output_filename)
        cv2.imwrite(output_filename, frame)

    print('Done!')

    # Save info in settings file if it doesn't exist already
    if not existing:

        with open(settings_filepath, 'a') as f:
            data_write_list = [vid_filepath,room,loc,initial_direction,end_direction,cut_start,cut_end,n_directions]

            # Construct string to write in
            data_write = '\n'
            for ind, item in enumerate(data_write_list):
                data_write += str(item)
                if ind != len(data_write_list) - 1:
                    data_write += ','

            print(f'Writing as: {data_write}')
            f.write(data_write)

if __name__ == '__main__':

    vid_filepath = 'data/n216_door/VID_20261012_125338.mp4' # Filepath to video (str)
    room = 'n216' # Room number (str)
    loc = 'door' # Location in the room (str)

    initial_direction = 0 # Initial direction that is being faced (deg)
    end_direction = 170 # End direction that is being faced (deg)

    cut_start = 0 # Starting time of video to cut (s)
    cut_end = 9 # End time of video to cut (-1 for end of video) (s)

    n_directions = 20 # How many images and directions to extract from the video (int)

    settings_filepath = 'make_dataset_settings.txt' # Store the settings used in here (leave it as 'make_dataset_settings.txt' probably)
    output_dir = 'data_train' # Big folder to save the data (leave it as 'data_train' probably)
        
    make_dataset_train(settings_filepath, 
                        vid_filepath, output_dir, 
                        room, loc, 
                        initial_direction, 
                        end_direction, 
                        cut_start, cut_end, 
                        n_directions)
