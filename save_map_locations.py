import matplotlib.pyplot as plt
import csv

# 1. Initialize an empty list to store coordinates
clicked_coords = []
settings_filepath = 'make_dataset_settings.txt' # Store the settings used in here (leave it as 'make_dataset_settings.txt' probably)
roomloc_filepath = 'roomloc_coords.txt'

# Get rooms list
room_list = []
loc_list = []

# Get the rooms and locations in the settings_filepath
with open(settings_filepath, mode='r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == 'vid_filepath':  # Get rid of the first row
            continue
        room = row[1]
        loc = row[2]

        room_list.append(room)
        loc_list.append(loc)

def onclick(event):

    # Check if click occurs in the axes
    if event.inaxes is not None:

        # Get the coordinates of the click
        x, y = event.xdata, event.ydata
        
        # Store coordinates of the click
        clicked_coords.append((int(x), int(y)))
        print(f"Recorded point: x={x}, y={y}")
        
        # Draw a point at the clicked location
        plt.plot(x, y, 'ro')
        event.canvas.draw_idle() # Rerender the plot

        # Stop after the last location
        if len(clicked_coords) == len(room_list):
            fig.canvas.mpl_disconnect(cid)
        else:
            # Change title to next location
            plt.title(f'{room_list[len(clicked_coords)]}, {loc_list[len(clicked_coords)]}')


# Set up the figure and axes
img = plt.imread('usyd_map.png')

fig, ax = plt.subplots()
plot_extent = [0, img.shape[1], 0, img.shape[0]] 
ax.imshow(img, extent=plot_extent, aspect='auto', alpha=0.6)
ax.set_title(f'{room_list[0]}, {loc_list[0]}')

# Click event
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Show plot
plt.show()

# Print all coordinates
print("\nAll captured coordinates:")
print(clicked_coords)

# Check that the number of locations matches the number of coordinates stored
if len(clicked_coords) != len(room_list):
    print('Lengths dont match')
    exit()

# Write the room locations to the filepath
with open(roomloc_filepath, 'w') as f:
    f_write = ''
    for i in range(len(clicked_coords)):
        f_write += f'{room_list[i]},{loc_list[i]},{clicked_coords[i][0]},{clicked_coords[i][1]}\n'

    f.write(f_write)