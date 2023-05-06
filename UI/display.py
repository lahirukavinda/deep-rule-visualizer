import tkinter as tk
from PIL import ImageTk, Image
import csv
import math

# Global file names
TEST_DIR, TRAIN_DIR, Y_TEST_FILE, DISTANCE_IMAGES_FILE, DISTANCES_FILE = '', '', '', '', ''

# Global variables
image_paths = []
resulted_images = []
distances = []

copy_image_paths = []
copy_resulted_images = []
copy_distances = []

incorrect_image_paths = []
incorrect_resulted_images = []
incorrect_distances = []

IMAGES_PER_COLUMN = 4


# Load data from files
def load_data():
    global image_paths, resulted_images, distances, \
        incorrect_image_paths, incorrect_resulted_images, incorrect_distances, \
        copy_image_paths, copy_resulted_images, copy_distances
    image_paths.clear()
    resulted_images.clear()
    distances.clear()

    with open(Y_TEST_FILE, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            image_name = row[0]  # row[0].split('.')[0]
            directory = image_name.split('_')[0]
            image_paths.append(TEST_DIR + directory + '/' + image_name)

    with open(DISTANCE_IMAGES_FILE, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        resulted_images = [row for row in reader]

    with open(DISTANCES_FILE, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        distances = [row for row in reader]

    for image_path, resulted_image, dis in zip(image_paths, resulted_images, distances):
        image_name = image_path.split('/')[-1]
        expected_name = image_name.split('_')[0]
        resulted_names = resulted_image[0].split('_')[0]  # [row.split('_')[0] for row in resulted_image if row]

        if expected_name != resulted_names:
            incorrect_image_paths.append(image_path)
            incorrect_resulted_images.append(resulted_image)
            incorrect_distances.append(dis)

        copy_image_paths.append(image_path)
        copy_resulted_images.append(resulted_image)
        copy_distances.append(dis)


def display_selected_image(image_listbox, test_image):
    # Get the selected image from the listbox
    selected_image = image_listbox.get(image_listbox.curselection())
    # Load and display the selected image
    image = Image.open(TEST_DIR + selected_image.split('_')[0] + '/' + selected_image)
    image = image.resize((300, 300), Image.ANTIALIAS)
    # image.thumbnail((400, 400))  # Resize the image to fit in the display area
    photo = ImageTk.PhotoImage(image)
    test_image.configure(image=photo)
    test_image.image = photo


# UI function to load and display images
def load_images(slider, image_listbox, canvas, frame, test_image, rule_count_label):
    if not image_listbox.curselection():
        return

    display_selected_image(image_listbox, test_image)

    # Display rule images
    selected_row = image_listbox.curselection()[0]

    selected_image_paths = resulted_images[selected_row]
    selected_image_paths = [row for row in selected_image_paths if row]

    selected_distances = distances[selected_row]
    selected_distances = [row for row in selected_distances if row]

    percentage = transform_log(float(slider.get())) / 100
    num_images = int(len(selected_image_paths) * percentage)

    selected_distances = [round(float(distance), 3) for distance in selected_distances[:num_images]]
    selected_image_paths = selected_image_paths[:num_images]

    # Clear the frame
    for widget in frame.winfo_children():
        widget.destroy()

    # Display images in grid layout
    row_count = (num_images + IMAGES_PER_COLUMN - 1) // IMAGES_PER_COLUMN
    for i in range(row_count):
        for j in range(IMAGES_PER_COLUMN):
            image_index = i * IMAGES_PER_COLUMN + j
            if image_index < num_images:
                image_path = selected_image_paths[image_index]
                distance = selected_distances[image_index]
                # Load and resize image
                img = Image.open(TRAIN_DIR + image_path.split('_')[0] + '/' + image_path)
                img = img.resize((100, 100), Image.ANTIALIAS)
                img = ImageTk.PhotoImage(img)
                # Create image label with distance
                label = tk.Label(frame, image=img)
                label.image = img
                label.grid(row=i * 2, column=j, padx=10, pady=10, sticky='w')
                distance_label = tk.Label(frame, text=f"Distance: {distance}")
                distance_label.grid(row=i * 2 + 1, column=j, padx=10, pady=0, sticky='w')

    # Update the right canvas scroll region
    canvas.update_idletasks()
    canvas.config(scrollregion=canvas.bbox('all'))

    # Update the number of rule images label
    rule_count_label.config(text=f"Showing {num_images} rule images")


def transform_log(value):
    # Convert the logarithmic value back to the original scale
    transformed_value = 10 ** value
    # Map the transformed value to the output range (0-100)
    output_value = (transformed_value - 1) / 9 * 100
    return output_value


def transform_exp(value):
    # Map the input value to the transformed logarithmic scale
    transformed_value = (value / 100 * 9) + 1
    # Apply the logarithmic transform to the transformed value
    transformed_value = math.log10(transformed_value)
    return transformed_value


def update_listbox(image_listbox, tick_box_var, num_images_label):
    global image_paths, resulted_images, distances

    # Clear the existing items in the Listbox
    image_listbox.delete(0, tk.END)

    # Get the state of the Tickbox
    is_correct = tick_box_var.get()

    # Populate the Listbox based on the Tickbox state
    if is_correct:
        # Update the Listbox with the list of correct images
        for item in incorrect_image_paths:
            image_listbox.insert(tk.END, item.split('/')[-1])
        image_paths = incorrect_image_paths
        resulted_images = incorrect_resulted_images
        distances = incorrect_distances
    else:
        for image_path in copy_image_paths:
            image_listbox.insert(tk.END, image_path.split('/')[-1])
        image_paths = copy_image_paths
        resulted_images = copy_resulted_images
        distances = copy_distances

    num_images_label.config(text=f"Listing {len(image_paths)} test images")


def UI_body():
    # Create the main window
    window = tk.Tk()
    window.title("DEEP RULE VISUALIZER")

    # Load the data
    load_data()

    # Component 1: Test image list box
    # Create a listbox to display image names
    image_listbox = tk.Listbox(window, height=24)
    for image_path in image_paths:
        image_listbox.insert(tk.END, image_path.split('/')[-1])
    image_listbox.place(x=22, y=101)

    # Component 2: Inorrectly classified Tick box
    # Create a tick box
    tick_box_label = tk.Label(window, text="Incorrectly classified images only")
    tick_box_label.grid(row=0, column=0, padx=(20, 0), pady=(20, 0), ipadx=2, ipady=1, sticky='nw')
    tick_box_var = tk.BooleanVar()
    tick_box = tk.Checkbutton(window, variable=tick_box_var)
    tick_box.place(x=230, y=20)

    # Component 3: Total number of test images label
    num_images_label = tk.Label(window, text="Listing 0 test images")
    num_images_label.place(x=22, y=60)

    # Component 4: Rule limit slider
    # Create a label for the slider
    slider_label = tk.Label(window, text="Logarithmic Scale Slider")
    slider_label.grid(row=5, column=0, padx=(20, 0), pady=20, sticky='w')

    rule_count_label = tk.Label(window, text="Showing 0 rule images")
    rule_count_label.grid(row=5, column=2, pady=20, columnspan=2)

    # Create the logarithmic scale slider
    slider = tk.Scale(window, from_=transform_exp(0), to=transform_exp(100), orient=tk.HORIZONTAL, resolution=0.1,
                      length=200)
    slider.set(transform_exp(2))
    slider.grid(row=5, column=1, pady=(0, 20), sticky='we')

    # Component 5: Selected test image viewer
    # Create a label to display the selected image
    test_image = tk.Label(window)
    test_image.grid(row=0, column=1, rowspan=4, padx=20)

    # Component 6: Rule images viewer
    # Create a canvas for the right side of the UI
    right_canvas = tk.Canvas(window, height=500, width=500)
    right_canvas.grid(row=0, column=2, rowspan=3, padx=20)

    # Component 7: Rule images scroll bar
    # Create a scrollbar for the right side of the UI
    right_scrollbar = tk.Scrollbar(window)
    right_scrollbar.grid(row=0, column=3, rowspan=1, sticky='ns')

    # Configure the canvas to work with the scrollbar
    right_canvas.config(yscrollcommand=right_scrollbar.set)
    right_scrollbar.config(command=right_canvas.yview)

    # Create a frame inside the right canvas to hold the images
    right_frame = tk.Frame(right_canvas)
    right_canvas.create_window((0, 0), window=right_frame, anchor='nw')

    # Function to update the right canvas scroll region
    def update_right_canvas_scroll_region(event):
        right_canvas.configure(scrollregion=right_canvas.bbox('all'))

    # Bind the update_right_canvas_scroll_region function to the right frame size change event
    right_frame.bind('<Configure>', update_right_canvas_scroll_region)

    # Bind the mouse wheel scroll event to the right canvas
    def on_mousewheel(event):
        right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    right_canvas.bind("<MouseWheel>", on_mousewheel)

    # Bind the load_images function to the Listbox selection event
    image_listbox.bind('<<ListboxSelect>>',
                       lambda event: load_images(slider, image_listbox, right_canvas, right_frame, test_image
                                                 , rule_count_label))

    # Bind the tickbox change event to the update_image_listbox function
    tick_box_var.trace('w', lambda *args: update_listbox(image_listbox, tick_box_var, num_images_label))

    # Run the main window loop
    window.mainloop()


def UI_init(data_set, data_set_directory):
    global TEST_DIR, TRAIN_DIR, Y_TEST_FILE, DISTANCE_IMAGES_FILE, DISTANCES_FILE
    if data_set == 'mnist':
        TEST_DIR = data_set_directory + 'test/'
        TRAIN_DIR = data_set_directory + 'train/'
    else:
        TEST_DIR = data_set_directory
        TRAIN_DIR = data_set_directory

    features_dir = f'features_{data_set}/'
    Y_TEST_FILE = features_dir + 'data_df_y_test.csv'

    results_dir = f'testdata_results_{data_set}/'
    DISTANCE_IMAGES_FILE = results_dir + 'dataset_distance_images.csv'
    DISTANCES_FILE = results_dir + 'dataset_distances.csv'


def UI(data_set, data_set_directory):
    UI_init(data_set, data_set_directory)
    UI_body()
