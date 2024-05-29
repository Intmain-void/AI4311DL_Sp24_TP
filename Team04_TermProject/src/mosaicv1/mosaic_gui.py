import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, Scrollbar
from PIL import Image, ImageTk
from detect_face import FaceDetector  # Assume this is your face detection module
import cv2
import os
import datetime
import face_recognition
import numpy as np


class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection and Mosaic Application")
        self.root.geometry("1280x800")
        self.root.iconbitmap("pyico.ico")

        self.detector = FaceDetector("best.pt")  # load your model
        self.image_path = None
        self.original_image = None
        self.current_image = None
        self.detected_image = None
        self.mosaic_image = None
        self.detection_boxes = []
        self.face_thumbnails = []
        self.registered_faces = []
        self.selected_registered_faces = []  # List to store selected registered faces
        self.selected_unregistered_faces = []  # List to store selected unregistered faces
        self.matched_faces = {}  # Dictionary to store matched faces between lists
        self.mosaic_factor = 1.0

        # Create main frames
        self.left_frame = tk.Frame(self.root)
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.right_frame = tk.Frame(self.root)
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

        # Left frame for face lists
        self.create_left_frame()

        # Right frame for image display
        self.create_right_frame()

        # Bottom frame for control buttons
        self.create_bottom_frame()

        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def create_left_frame(self):
        # Registered face list
        self.registered_frame = tk.LabelFrame(
            self.left_frame, text="Registered Detected Face List"
        )
        self.registered_frame.pack(fill="both", expand=True)

        self.registered_canvas = Canvas(self.registered_frame)
        self.registered_scrollbar = Scrollbar(
            self.registered_frame,
            orient="vertical",
            command=self.registered_canvas.yview,
        )
        self.registered_scrollable_frame = tk.Frame(self.registered_canvas)

        self.registered_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.registered_canvas.configure(
                scrollregion=self.registered_canvas.bbox("all")
            ),
        )

        self.registered_canvas.create_window(
            (0, 0), window=self.registered_scrollable_frame, anchor="nw"
        )
        self.registered_canvas.configure(yscrollcommand=self.registered_scrollbar.set)

        self.registered_canvas.pack(side="left", fill="both", expand=True)
        self.registered_scrollbar.pack(side="right", fill="y")

        self.registered_select_all = tk.Button(
            self.registered_frame, text="Select All", command=self.select_all_registered
        )
        self.registered_select_all.pack(side="top", fill="x")
        self.registered_select_none = tk.Button(
            self.registered_frame,
            text="Select None",
            command=self.deselect_all_registered,
        )
        self.registered_select_none.pack(side="top", fill="x")

        # Unregistered face list
        self.unregistered_frame = tk.LabelFrame(
            self.left_frame, text="Unregistered Detected Face List"
        )
        self.unregistered_frame.pack(fill="both", expand=True)

        self.unregistered_canvas = Canvas(self.unregistered_frame)
        self.unregistered_scrollbar = Scrollbar(
            self.unregistered_frame,
            orient="vertical",
            command=self.unregistered_canvas.yview,
        )
        self.unregistered_scrollable_frame = tk.Frame(self.unregistered_canvas)

        self.unregistered_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.unregistered_canvas.configure(
                scrollregion=self.unregistered_canvas.bbox("all")
            ),
        )

        self.unregistered_canvas.create_window(
            (0, 0), window=self.unregistered_scrollable_frame, anchor="nw"
        )
        self.unregistered_canvas.configure(
            yscrollcommand=self.unregistered_scrollbar.set
        )

        self.unregistered_canvas.pack(side="left", fill="both", expand=True)
        self.unregistered_scrollbar.pack(side="right", fill="y")

        self.unregistered_select_all = tk.Button(
            self.unregistered_frame,
            text="Select All",
            command=self.select_all_unregistered,
        )
        self.unregistered_select_all.pack(side="top", fill="x")
        self.unregistered_select_none = tk.Button(
            self.unregistered_frame,
            text="Select None",
            command=self.deselect_all_unregistered,
        )
        self.unregistered_select_none.pack(side="top", fill="x")

    def create_right_frame(self):
        self.image_label = tk.Label(self.right_frame, text="Image")
        self.image_label.pack(fill="both", expand=True)
        self.image_label.bind("<Configure>", self.resize_image)

    def create_bottom_frame(self):
        self.load_image_button = tk.Button(
            self.bottom_frame, text="Load Image", command=self.load_image
        )
        self.load_image_button.pack(side="left")

        self.load_registered_faces_button = tk.Button(
            self.bottom_frame,
            text="Load Registered Faces",
            command=self.load_registered_faces,
        )
        self.load_registered_faces_button.pack(side="left")

        self.detect_faces_button = tk.Button(
            self.bottom_frame, text="Detect Faces", command=self.detect_faces
        )
        self.detect_faces_button.pack(side="left")

        self.draw_mosaic_button = tk.Button(
            self.bottom_frame, text="Draw Mosaic", command=self.draw_mosaic
        )
        self.draw_mosaic_button.pack(side="left")

        self.intensity_label = tk.Label(self.bottom_frame, text="Intensity")
        self.intensity_label.pack(side="left")

        self.intensity_scale = tk.Scale(
            self.bottom_frame, from_=1, to=8, orient=tk.HORIZONTAL
        )
        self.intensity_scale.pack(side="left")

        self.export_button = tk.Button(
            self.bottom_frame, text="Export", command=self.export_image
        )
        self.export_button.pack(side="left")

    def select_all_registered(self):
        for widget in self.registered_scrollable_frame.winfo_children():
            if isinstance(widget, tk.Checkbutton):
                widget.select()

    def deselect_all_registered(self):
        for widget in self.registered_scrollable_frame.winfo_children():
            if isinstance(widget, tk.Checkbutton):
                widget.deselect()

    def select_all_unregistered(self):
        for widget in self.unregistered_scrollable_frame.winfo_children():
            if isinstance(widget, tk.Checkbutton):
                widget.select()

    def deselect_all_unregistered(self):
        for widget in self.unregistered_scrollable_frame.winfo_children():
            if isinstance(widget, tk.Checkbutton):
                widget.deselect()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)

    def load_registered_faces(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            print("Registered faces loaded from:", folder_path)
            self.scan_faces_in_folder(folder_path)
        else:
            messagebox.showerror("Error", "Cannot read folder path")

    def scan_faces_in_folder(self, folder_path):
        self.registered_faces.clear()
        for widget in self.registered_scrollable_frame.winfo_children():
            widget.destroy()

        messagebox.showinfo(
            "Start", "Start scanning faces from the given image folder."
        )
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                image_path = os.path.join(folder_path, filename)
                temp_bounding_boxes = self.detector.detect_faces(image_path)

                if not temp_bounding_boxes:
                    continue

                for box in temp_bounding_boxes:
                    x, y, w, h, embedding = box
                    if embedding is not None:
                        face_info = {
                            "file": image_path,
                            "bounding_box": (x, y, w, h),
                            "embedding": embedding,
                        }
                        self.registered_faces.append(face_info)
                        self.add_thumbnail(
                            self.registered_scrollable_frame,
                            face_info,
                            "registered",
                            len(self.registered_faces) - 1,
                        )

        messagebox.showinfo("End", "End scanning faces")
        if not self.registered_faces:
            messagebox.showwarning(
                "Warning", "No faces detected in the selected folder"
            )

        # Compare registered faces with unregistered faces
        self.compare_faces()

    def compare_faces(self):
        if not self.registered_faces or not self.detection_boxes:
            print("No faces to compare.")
            return

        registered_encodings = [
            face["embedding"]
            for face in self.registered_faces
            if face["embedding"] is not None
        ]
        unregistered_encodings = [
            box[4] for box in self.detection_boxes if box[4] is not None
        ]

        matches = 0
        self.matched_faces = {}  # Reset matched faces
        for unreg_index, unreg_enc in enumerate(unregistered_encodings):
            match = face_recognition.compare_faces(
                registered_encodings, unreg_enc, tolerance=0.55
            )
            for reg_index, same in enumerate(match):
                if same:
                    self.matched_faces.setdefault(reg_index, []).append(unreg_index)
                    matches += 1

        print(f"Number of face matches: {matches}")
        print(f"Matched Faces: {self.matched_faces}")

    def detect_faces(self):
        if self.image_path is None:
            messagebox.showerror("Error", "Please load an image first.")
            return

        self.detection_boxes = self.detector.detect_faces(self.image_path)
        messagebox.showinfo("Info", "Face Detection ended")

        self.update_unregistered_list()

    def update_unregistered_list(self):
        for widget in self.unregistered_scrollable_frame.winfo_children():
            widget.destroy()
        self.face_thumbnails.clear()
        for i, box in enumerate(self.detection_boxes):
            x, y, w, h, emb = box
            face_info = {
                "bounding_box": (x, y, w, h),
                "embedding": emb,
            }
            self.face_thumbnails.append(face_info)
            self.add_thumbnail(
                self.unregistered_scrollable_frame, face_info, "unregistered", i
            )

    def add_thumbnail(self, frame, face_info, list_type, index=None):
        grid_size = 60  # Define the grid size (60x60 pixels)
        thumbnail_size = 50  # Define the thumbnail size (50x50 pixels)

        x, y, w, h = face_info["bounding_box"]
        if list_type == "unregistered":
            img = Image.open(self.image_path).crop((x, y, x + w, y + h))
        else:
            img = Image.open(face_info["file"]).crop((x, y, x + w, y + h))
        img.thumbnail((thumbnail_size, thumbnail_size))
        photo = ImageTk.PhotoImage(img)

        btn = tk.Checkbutton(frame, image=photo)
        btn.image = photo
        btn.var = tk.BooleanVar()
        btn.config(variable=btn.var)
        btn.var.set(False)
        btn.var.trace(
            "w",
            lambda *args, idx=index, var=btn.var, lst_type=list_type: self.on_thumbnail_click(
                idx, var, lst_type
            ),
        )

        num_children = len(frame.grid_slaves())
        row = num_children // 3
        col = num_children % 3
        btn.grid(
            row=row,
            column=col,
            ipadx=(grid_size - thumbnail_size) // 2,
            ipady=(grid_size - thumbnail_size) // 2,
        )

    def on_thumbnail_click(self, index, var, list_type):
        if list_type == "unregistered":
            if var.get():
                if index not in self.selected_unregistered_faces:
                    self.selected_unregistered_faces.append(index)
                # Sync selection with matched registered faces
                for reg_index, unreg_indices in self.matched_faces.items():
                    if index in unreg_indices:
                        widget = self.registered_scrollable_frame.grid_slaves(
                            row=reg_index // 3, column=reg_index % 3
                        )[0]
                        widget.select()
            else:
                if index in self.selected_unregistered_faces:
                    self.selected_unregistered_faces.remove(index)
                # Sync deselection with matched registered faces
                for reg_index, unreg_indices in self.matched_faces.items():
                    if index in unreg_indices:
                        widget = self.registered_scrollable_frame.grid_slaves(
                            row=reg_index // 3, column=reg_index % 3
                        )[0]
                        widget.deselect()
        elif list_type == "registered":
            if var.get():
                if index not in self.selected_registered_faces:
                    self.selected_registered_faces.append(index)
                # Sync selection with matched unregistered faces
                if index in self.matched_faces:
                    for unreg_index in self.matched_faces[index]:
                        widget = self.unregistered_scrollable_frame.grid_slaves(
                            row=unreg_index // 3, column=unreg_index % 3
                        )[0]
                        widget.select()
            else:
                if index in self.selected_registered_faces:
                    self.selected_registered_faces.remove(index)
                # Sync deselection with matched unregistered faces
                if index in self.matched_faces:
                    for unreg_index in self.matched_faces[index]:
                        widget = self.unregistered_scrollable_frame.grid_slaves(
                            row=unreg_index // 3, column=unreg_index % 3
                        )[0]
                        widget.deselect()

        self.update_image_with_boxes()

    def update_image_with_boxes(self):
        if self.image_path is None:
            return
        image = cv2.imread(self.image_path)
        if image is None:
            return

        for i in self.selected_unregistered_faces:
            if i < len(self.detection_boxes):
                x, y, w, h, emb = self.detection_boxes[i]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.show_image(image)

    def draw_mosaic(self):
        if self.image_path is None:
            messagebox.showerror("Error", "Please load an image first.")
            return
        image = cv2.imread(self.image_path)
        if image is None:
            messagebox.showerror("Error", "Could not read the image file.")
            return

        mosaic_intensity = self.intensity_scale.get()

        for i in self.selected_unregistered_faces:
            if i < len(self.detection_boxes):
                x, y, w, h, emb = self.detection_boxes[i]
                face_image = image[y : y + h, x : x + w]
                # Increase mosaic intensity by making the pixelation more coarse
                face_image = cv2.resize(
                    face_image,
                    (
                        max(1, w // (mosaic_intensity * 2)),
                        max(1, h // (mosaic_intensity * 2)),
                    ),
                )
                face_image = cv2.resize(
                    face_image, (w, h), interpolation=cv2.INTER_NEAREST
                )
                image[y : y + h, x : x + w] = face_image

        self.mosaic_image = image
        self.show_image(image)

    def export_image(self):
        if self.mosaic_image is None:
            messagebox.showerror("Error", "No mosaiced image to export.")
            return

        export_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")],
            initialfile=os.path.basename(self.image_path).split(".")[0]
            + "_mosaic_"
            + datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
        )
        if export_path:
            cv2.imwrite(export_path, self.mosaic_image)
            messagebox.showinfo("Success", f"Image saved to {export_path}")

    def display_image(self, file_path):
        image = Image.open(file_path)
        if image is not None:
            self.original_image = image
            self.update_image_label(image)
        else:
            messagebox.showerror("Error", "Failed to load image.")

    def resize_image(self, event):
        if hasattr(self, "original_image") and self.original_image is not None:
            self.update_image_label(self.original_image)

    def update_image_label(self, image):
        if image is None:
            return

        width, height = self.image_label.winfo_width(), self.image_label.winfo_height()
        if width == 0 or height == 0:
            return

        aspect_ratio = image.width / image.height
        if width / aspect_ratio <= height:
            new_width = width
            new_height = int(width / aspect_ratio)
        else:
            new_height = height
            new_width = int(height * aspect_ratio)

        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=self.photo, text="")

    def show_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        self.update_image_label(image_pil)


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
