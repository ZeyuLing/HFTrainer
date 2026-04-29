# import os
# import vedo
# import sys
# import time

# class ModelViewer:
#     def __init__(self, obj_files, highlight_frames=None, fps=20):
#         self.obj_files = obj_files
#         self.fps = fps
#         self.current_frame = 0
#         self.playing = True
#         self.scene = vedo.Plotter()  # Interactive mode enabled by default
#         self.last_model = None
#         self.frame_interval = 1.0 / fps  # Time in seconds for frame rate control
#         self.light = vedo.Light(pos=(0, 1000, 1000), focal_point=(0, 0, 0), intensity=1)
#         self.scene += self.light
#         self.last_time = time.time()

#         # Determine ground plane based on the first frame's lowest Y point
#         self.ground_plane_y = self.get_ground_plane_y(obj_files[0])

#         # Create a ground plane in the XZ plane at the lowest Y
#         self.ground_plane = vedo.Plane(pos=(0, self.ground_plane_y, 0), normal=(0, 1, 0), s=(10, 10), c="lightgrey", alpha=0.5)
#         self.scene += self.ground_plane

#         # Add a grid, rotate it, and then move it to the correct Y position
#         self.grid = vedo.Grid(s=(10, 10), res=(100, 100), c="black", alpha=0.2)
#         self.grid.rotate_x(90)  # Rotate to align with the XZ plane (Y up)
#         self.grid.pos(0, self.ground_plane_y, 0)  # Now position it at the correct Y level
#         self.scene += self.grid

#         # Highlight frames from the provided file (if any)
#         self.highlight_frames = highlight_frames if highlight_frames is not None else []
#         self.highlight_models = self.load_highlight_models()  # Load the models for highlight frames

#     def get_ground_plane_y(self, obj_file):
#         # Load the first OBJ file
#         model = vedo.load(obj_file)
#         if model is None:
#             print(f"Failed to load {obj_file}")
#             return 0

#         # Get all vertices and find the minimum Y-coordinate
#         vertices = model.points()
#         min_y = min([v[1] for v in vertices])  # v[1] corresponds to Y-coordinate

#         print(f"Lowest Y-coordinate in {obj_file}: {min_y}")
#         return min_y

#     def load_highlight_models(self):
#         highlight_models = []
#         for frame in self.highlight_frames:
#             if 0 <= frame < len(self.obj_files):  # Check if the frame index is valid
#                 model = vedo.load(self.obj_files[frame])
#                 if model is not None:
#                     model.color("red").alpha(0.1)  # Apply color and transparency to highlight models
#                     model.pos(0, 0, 0)
#                     highlight_models.append(model)
#         return highlight_models

#     def load_and_display_frame(self):
#         # Remove previous model
#         if self.last_model:
#             self.scene.remove(self.last_model)

#         # Load current model
#         obj_file = self.obj_files[self.current_frame]
#         model = vedo.load(obj_file)

#         if model is None:
#             print(f"Failed to load {obj_file}")
#             return

#         # Set model position and display
#         model.pos(0, 0, 0)

#         # Display the current model
#         self.scene += model

#         # Update the reference to the last model
#         self.last_model = model
#         print(f"Displaying: {obj_file}, Frame: {self.current_frame}")

#         # Force render the scene to update the display
#         self.scene.render()

#     def display_highlight_frames(self):
#         """Ensure that highlight frames are always visible in the scene."""
#         for model in self.highlight_models:
#             if model not in self.scene.actors:
#                 self.scene += model  # Add the highlight model to the scene

#     def update(self, event=None):
#         current_time = time.time()
#         elapsed_time = current_time - self.last_time

#         if self.playing and elapsed_time >= self.frame_interval:
#             self.current_frame = (self.current_frame + 1) % len(self.obj_files)
#             self.load_and_display_frame()
#             self.display_highlight_frames()  # Always show highlight frames
#             self.last_time = current_time

#     def toggle_play_pause(self):
#         self.playing = not self.playing
#         print("Paused" if not self.playing else "Playing")

#     def next_frame(self):
#         self.playing = False
#         self.current_frame = (self.current_frame + 1) % len(self.obj_files)
#         self.load_and_display_frame()
#         self.display_highlight_frames()  # Ensure highlight frames are displayed
#         print("Next frame.")

#     def previous_frame(self):
#         self.playing = False
#         self.current_frame = (self.current_frame - 1) % len(self.obj_files)
#         self.load_and_display_frame()
#         self.display_highlight_frames()  # Ensure highlight frames are displayed
#         print("Previous frame.")

#     def restart(self):
#         self.playing = True
#         self.current_frame = 0
#         print("Restarting playback.")
#         self.load_and_display_frame()
#         self.display_highlight_frames()  # Ensure highlight frames are displayed

#     def handle_keypress(self, event):
#         key = event.keypress  # Correct way to access keypress in vedo

#         if key == "space":
#             self.toggle_play_pause()
#         elif key == "Right":
#             self.next_frame()
#         elif key == "Left":
#             self.previous_frame()
#         elif key == "r":
#             self.restart()

# def load_and_sort_obj_files(directory):
#     # Get all .obj files in the specified directory
#     obj_files = [f for f in os.listdir(directory) if f.endswith('.obj')]
    
#     # Sort files by name
#     obj_files.sort()
    
#     # Create a list with the full path of each file
#     obj_paths = [os.path.join(directory, f) for f in obj_files]
    
#     return obj_paths

# def load_highlight_frames(file_path):
#     """
#     Load frames to highlight from a given text file.
#     Each line of the file should contain a frame index (0-based).
#     """
#     if not os.path.isfile(file_path):
#         print(f"Highlight file {file_path} does not exist. No frames will be highlighted.")
#         return []

#     with open(file_path, 'r') as f:
#         highlight_frames = [int(line.strip()) for line in f.readlines()]
    
#     return highlight_frames

# def visualize_obj_files(directory, highlight_file=None, fps=20):
#     # Load and sort OBJ files
#     obj_files = load_and_sort_obj_files(directory)
    
#     if not obj_files:
#         print("No OBJ files found in the directory.")
#         return

#     # Load highlight frames if the file is provided
#     highlight_frames = load_highlight_frames(highlight_file) if highlight_file else []

#     # Create the viewer instance
#     viewer = ModelViewer(obj_files, highlight_frames, fps)

#     # Set up keyboard controls
#     viewer.scene.add_callback("key press", viewer.handle_keypress)

#     # Add a timer callback for frame updates
#     viewer.scene.add_callback("timer", viewer.update)
#     viewer.scene.timer_callback("start")  # Start timer with default interval

#     # Start the interactive window
#     viewer.scene.show(interactive=True)

# if __name__ == "__main__":
#     # Check if the directory argument is provided
#     if len(sys.argv) < 2:
#         print("Please provide the path to a directory containing OBJ files.")
#         sys.exit(1)
    
#     # Get the directory path from the command-line argument
#     obj_directory = sys.argv[1]
    
#     # Optional: Get the highlight file path (if provided)
#     highlight_file = sys.argv[2] if len(sys.argv) > 2 else None

#     # Ensure the directory exists
#     if not os.path.isdir(obj_directory):
#         print(f"The directory {obj_directory} does not exist. Please check the path.")
#         sys.exit(1)
    
#     # Visualize the OBJ files with playback controls and frame highlighting
#     visualize_obj_files(obj_directory, highlight_file, fps=20)

import os
import vedo
import sys
import time

class ModelViewer:
    def __init__(self, obj_files, highlight_frames=None, fps=20):
        self.obj_files = obj_files
        self.fps = fps
        self.current_frame = 0
        self.playing = True
        self.scene = vedo.Plotter()  # Interactive mode enabled by default
        self.last_model = None
        self.frame_interval = 1.0 / fps  # Time in seconds for frame rate control
        self.light = vedo.Light(pos=(0, 1000, 1000), focal_point=(0, 0, 0), intensity=1)
        self.scene += self.light
        self.last_time = time.time()

        # Determine ground plane based on the first frame's lowest Y point
        self.ground_plane_y = self.get_ground_plane_y(obj_files[0])

        # Create a ground plane in the XZ plane at the lowest Y
        self.ground_plane = vedo.Plane(pos=(0, self.ground_plane_y, 0), normal=(0, 1, 0), s=(10, 10), c="lightgrey", alpha=0.5)
        self.scene += self.ground_plane

        # Add a grid, rotate it, and then move it to the correct Y position
        self.grid = vedo.Grid(s=(10, 10), res=(100, 100), c="black", alpha=0.2)
        self.grid.rotate_x(90)  # Rotate to align with the XZ plane (Y up)
        self.grid.pos(0, self.ground_plane_y, 0)  # Now position it at the correct Y level
        self.scene += self.grid

        # Highlight frames from the provided file (if any)
        self.highlight_frames = highlight_frames if highlight_frames is not None else []

    def get_ground_plane_y(self, obj_file):
        # Load the first OBJ file
        model = vedo.load(obj_file)
        if model is None:
            print(f"Failed to load {obj_file}")
            return 0

        # Get all vertices and find the minimum Y-coordinate
        vertices = model.points()
        min_y = min([v[1] for v in vertices])  # v[1] corresponds to Y-coordinate

        print(f"Lowest Y-coordinate in {obj_file}: {min_y}")
        return min_y

    def load_and_display_frame(self):
        # Remove previous models (current, previous highlight, and next highlight)
        if self.last_model:
            self.scene.remove(self.last_model)
        if hasattr(self, 'prev_highlight_model') and self.prev_highlight_model:
            self.scene.remove(self.prev_highlight_model)
        if hasattr(self, 'next_highlight_model') and self.next_highlight_model:
            self.scene.remove(self.next_highlight_model)

        # Load current model
        obj_file = self.obj_files[self.current_frame]
        model = vedo.load(obj_file)

        if model is None:
            print(f"Failed to load {obj_file}")
            return

        # Set model position and display
        model.pos(0, 0, 0)
        self.scene += model

        # Update the reference to the last model
        self.last_model = model
        print(f"Displaying: {obj_file}, Frame: {self.current_frame}")

        # Find and display previous and next highlight frames
        self.display_highlight_frames()

        # Force render the scene to update the display
        self.scene.render()

    def display_highlight_frames(self):
        """Display the previous and next highlight frames."""
        prev_highlight_frame = None
        next_highlight_frame = None

        # Find the previous and next highlight frames relative to the current frame
        for frame in self.highlight_frames:
            if frame < self.current_frame:
                prev_highlight_frame = frame
            elif frame > self.current_frame:
                next_highlight_frame = frame
                break

        # Display previous highlight frame in blue
        if prev_highlight_frame is not None:
            prev_obj_file = self.obj_files[prev_highlight_frame]
            self.prev_highlight_model = vedo.load(prev_obj_file)
            if self.prev_highlight_model is not None:
                self.prev_highlight_model.color("blue").alpha(0.5)
                self.prev_highlight_model.pos(0, 0, 0)
                self.scene += self.prev_highlight_model

        # Display next highlight frame in red
        if next_highlight_frame is not None:
            next_obj_file = self.obj_files[next_highlight_frame]
            self.next_highlight_model = vedo.load(next_obj_file)
            if self.next_highlight_model is not None:
                self.next_highlight_model.color("red").alpha(0.5)
                self.next_highlight_model.pos(0, 0, 0)
                self.scene += self.next_highlight_model

    def update(self, event=None):
        current_time = time.time()
        elapsed_time = current_time - self.last_time

        if self.playing and elapsed_time >= self.frame_interval:
            self.current_frame = (self.current_frame + 1) % len(self.obj_files)
            self.load_and_display_frame()
            self.last_time = current_time

    def toggle_play_pause(self):
        self.playing = not self.playing
        print("Paused" if not self.playing else "Playing")

    def next_frame(self):
        self.playing = False
        self.current_frame = (self.current_frame + 1) % len(self.obj_files)
        self.load_and_display_frame()
        print("Next frame.")

    def previous_frame(self):
        self.playing = False
        self.current_frame = (self.current_frame - 1) % len(self.obj_files)
        self.load_and_display_frame()
        print("Previous frame.")

    def restart(self):
        self.playing = True
        self.current_frame = 0
        print("Restarting playback.")
        self.load_and_display_frame()

    def handle_keypress(self, event):
        key = event.keypress  # Correct way to access keypress in vedo

        if key == "space":
            self.toggle_play_pause()
        elif key == "Right":
            self.next_frame()
        elif key == "Left":
            self.previous_frame()
        elif key == "r":
            self.restart()

def load_and_sort_obj_files(directory):
    # Get all .obj files in the specified directory
    obj_files = [f for f in os.listdir(directory) if f.endswith('.obj')]
    
    # Sort files by name
    obj_files.sort()
    
    # Create a list with the full path of each file
    obj_paths = [os.path.join(directory, f) for f in obj_files]
    
    return obj_paths

def load_highlight_frames(file_path):
    """
    Load frames to highlight from a given text file.
    Each line of the file should contain a frame index (0-based).
    """
    if not os.path.isfile(file_path):
        print(f"Highlight file {file_path} does not exist. No frames will be highlighted.")
        return []

    with open(file_path, 'r') as f:
        highlight_frames = [int(line.strip()) for line in f.readlines()]
    
    return highlight_frames

def visualize_obj_files(directory, highlight_file=None, fps=20):
    # Load and sort OBJ files
    obj_files = load_and_sort_obj_files(directory)
    
    if not obj_files:
        print("No OBJ files found in the directory.")
        return

    # Load highlight frames if the file is provided
    highlight_frames = load_highlight_frames(highlight_file) if highlight_file else []

    # Create the viewer instance
    viewer = ModelViewer(obj_files, highlight_frames, fps)

    # Set up keyboard controls
    viewer.scene.add_callback("key press", viewer.handle_keypress)

    # Add a timer callback for frame updates
    viewer.scene.add_callback("timer", viewer.update)
    viewer.scene.timer_callback("start")  # Start timer with default interval

    # Start the interactive window
    viewer.scene.show(interactive=True)

if __name__ == "__main__":
    # Check if the directory argument is provided
    if len(sys.argv) < 2:
        print("Please provide the path to a directory containing OBJ files.")
        sys.exit(1)
    
    # Get the directory path from the command-line argument
    obj_directory = sys.argv[1]
    
    # Optional: Get the highlight file path (if provided)
    highlight_file = sys.argv[2] if len(sys.argv) > 2 else None

    # Ensure the directory exists
    if not os.path.isdir(obj_directory):
        print(f"The directory {obj_directory} does not exist. Please check the path.")
        sys.exit(1)
    
    # Visualize the OBJ files with playback controls and frame highlighting
    visualize_obj_files(obj_directory, highlight_file, fps=20)
