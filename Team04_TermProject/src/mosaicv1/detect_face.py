from ultralytics import YOLO
import torch
import cv2
import numpy as np
import face_recognition


class FaceDetector:
    def __init__(self, model_path):
        # Check device type
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
            print("Using CUDA device")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print("Using MPS device")
        else:
            print("Using CPU device")

        # Load model and move it to the specified device
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def detect_faces(self, image_path, conf_threshold=0.25):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read the image at {image_path}")

        # Perform inference
        try:
            results = self.model(image)
        except Exception as e:
            print(f"Error during inference: {e}")
            raise

        detection_boxes = []
        for result in results:
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                if conf >= conf_threshold:
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    face_image = image[y : y + h, x : x + w]
                    face_embedding = self.get_face_embedding(face_image)
                    detection_boxes.append((x, y, w, h, face_embedding))

        return detection_boxes

    def get_face_embedding(self, face_image):
        # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        # Get face embeddings using face_recognition
        embeddings = face_recognition.face_encodings(rgb_face)
        # Return the first embedding found (there should be only one face per cropped face image)
        if embeddings:
            return embeddings[0]
        else:
            return None


# Example usage
if __name__ == "__main__":
    model_path = "../Pretrained yolov8face/yolov8m-face.pt"
    image_path = "../SampleData/Pedestrian.jpg"

    try:
        detector = FaceDetector(model_path)
        bounding_boxes = detector.detect_faces(image_path)
        for bbox in bounding_boxes:
            x, y, w, h, embedding = bbox
            print(f"Bounding Box: {x, y, w, h}, Embedding: {embedding}")
    except Exception as e:
        print(f"Error in face detection: {e}")


# ------------
