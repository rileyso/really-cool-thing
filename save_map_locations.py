import matplotlib.pyplot as plt
import csv

# 1. Initialize an empty list to store coordinates
clicked_coords = []
settings_filepath = 'make_dataset_settings.txt' # Store the settings used in here (leave it as 'make_dataset_settings.txt' probably)
roomloc_filepath = 'roomloc_coords.txt'

# Get rooms list
room_list = []
loc_list = []

with open(settings_filepath, mode='r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == 'vid_filepath':
            continue # Get rid of the first row
        room = row[1]
        loc = row[2]

        room_list.append(room)
        loc_list.append(loc)
        print(room, loc)


def onclick(event):
    # 2. Check if the click occurred inside the plot axes area
    if event.inaxes is not None:
        # Extract data-space coordinates
        x, y = event.xdata, event.ydata
        
        # 3. Store the coordinates
        clicked_coords.append((int(x), int(y)))
        print(f"Recorded point: x={x}, y={y}")
        
        # 4. Optional: Draw a red dot at the clicked location
        plt.plot(x, y, 'ro')
        event.canvas.draw_idle()  # Re-render the plot efficiently

        if len(clicked_coords) == len(room_list):
            fig.canvas.mpl_disconnect(cid)
        else:
            # Change title to next one
            plt.title(f'{room_list[len(clicked_coords)]}, {loc_list[len(clicked_coords)]}')


# Set up the figure and axes
img = plt.imread('usyd_map.png')
print(img.shape)

fig, ax = plt.subplots()
plot_extent = [0, img.shape[1], 0, img.shape[0]] 
ax.imshow(img, extent=plot_extent, aspect='auto', alpha=0.6)
ax.set_title(f'{room_list[0]}, {loc_list[0]}')

# 5. Connect the click event to your function
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

# Once the window is closed, your list is fully populated and ready to use
print("\nAll captured coordinates:")
print(clicked_coords)

if len(clicked_coords) != len(room_list):
    print('Lengths dont match')
    exit()

with open(roomloc_filepath, 'w') as f:
    f_write = ''
    for i in range(len(clicked_coords)):
        f_write += f'{room_list[i]},{loc_list[i]},{clicked_coords[i][0]},{clicked_coords[i][1]}\n'

    f.write(f_write)