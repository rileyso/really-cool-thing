import matplotlib.pyplot as plt
import csv
import numpy as np

def view_map_location(roomloc_filepath, room, loc, angle):

    # Clean inputs
    room = str(room)
    loc = str(loc)
    angle = int(angle)

    # Find the room location in roomloc
    found_roomloc = False
    with open(roomloc_filepath, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if (row[0] == room) and (row[1] == loc):
                x = int(row[2])
                y = int(row[3])
                found_roomloc = True
                break

        if not found_roomloc:
            raise ValueError(f'Room/loc pair ({room}, {loc}) not found.')

    # Plot map
    img = plt.imread('usyd_map.png')
    fig, ax = plt.subplots()
    plot_extent = [0, img.shape[1], 0, img.shape[0]] 
    ax.imshow(img, extent=plot_extent, aspect='auto', alpha=0.6)

    # Annotate with arrow
    amp = 5
    dx, dy = amp * np.sin(np.deg2rad(angle)), amp * np.cos(np.deg2rad(angle))
    end_pos = (x + dx, y + dy)

    ax.annotate(
        xy = end_pos,  # Arrow tip points here
        xytext = (x, y),  # Arrow tail starts here
        text = "",
        arrowprops=dict(
            color='red',
            arrowstyle='->',
            lw=2
       )
    )

    plt.show()

if __name__ == '__main__':

    roomloc_filepath = 'roomloc_coords.txt'
    filename = r'data_train\n203_door\n203_door_im1_101.jpg'

    split = filename.split('\\')[-1]
    split = split.split('/')[-1]
    split = split.split('.')[0]
    split = split.split('_')

    view_map_location(roomloc_filepath, split[0], split[1], split[3])