# import cv2
# import numpy as np
# import torch
# from insightface.app import FaceAnalysis
# import pickle
# import mediapipe as mp
#
# # Load SVM model
# with open("face_recognizer.pkl", "rb") as f:
#     svm = pickle.load(f)
#
# # Initialize InsightFace (ArcFace) for embeddings
# face_app = FaceAnalysis(providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
# face_app.prepare(ctx_id=0, det_size=(640, 640))
#
# # Initialize MediaPipe for face detection (consistent with training)
# mp_face_detection = mp.solutions.face_detection
# face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.3)
#
# # Initialize webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()
#
# # Confidence threshold for "Unknown" detection
# CONFIDENCE_THRESHOLD = 0.7  # Adjust as needed (0.0 to 1.0)
#
#
# def preprocess_image(image):
#     """Detect and crop faces using MediaPipe, resize for ArcFace embedding."""
#     rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_detector.process(rgb_frame)
#     if not results.detections:
#         return []
#
#     faces_data = []
#     h, w, _ = image.shape
#
#     # Process all detected faces
#     for detection in results.detections:
#         bbox = detection.location_data.relative_bounding_box
#
#         x_min = int(bbox.xmin * w)
#         y_min = int(bbox.ymin * h)
#         box_width = int(bbox.width * w)
#         box_height = int(bbox.height * h)
#         x_max = x_min + box_width
#         y_max = y_min + box_height
#
#         # Use expanded area for cropping (for embedding) but keep original bbox for display
#         expand_top, expand_side, expand_bottom = 1.0, 0.25, 0.2
#         crop_x_min = max(0, int(x_min - expand_side * box_width))
#         crop_y_min = max(0, int(y_min - expand_top * box_height))
#         crop_x_max = min(w, int(x_min + (1 + 2 * expand_side) * box_width))
#         crop_y_max = min(h, int(y_min + (1 + expand_top + expand_bottom) * box_height))
#
#         face_crop = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
#         if face_crop.size == 0:
#             continue
#
#         # Resize to match training input (224x224)
#         face_resized = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)
#         faces_data.append({
#             'image': cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB),
#             'bbox': (x_min, y_min, x_max, y_max)  # Original tight bbox for display
#         })
#
#     return faces_data
#
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture frame.")
#         break
#
#     # Preprocess frame for all faces
#     faces_data = preprocess_image(frame)
#
#     if faces_data:
#         for face_data in faces_data:
#             preprocessed_img = face_data['image']
#             bbox = face_data['bbox']
#             try:
#                 # Extract ArcFace embedding
#                 faces = face_app.get(preprocessed_img)
#                 if not faces:
#                     print("No embedding extracted for a face.")
#                     continue
#                 embedding = faces[0].embedding  # 512D
#                 embedding = embedding.reshape(1, -1)
#
#                 # Predict with SVM
#                 pred_name = svm.predict(embedding)[0]
#                 pred_prob = svm.predict_proba(embedding)[0].max()
#
#                 # Apply confidence threshold for "Unknown"
#                 if pred_prob >= CONFIDENCE_THRESHOLD:
#                     label = f"{pred_name} ({pred_prob:.2f})"
#                     color = (0, 255, 0)  # Green for known persons
#                 else:
#                     label = f"Unknown ({pred_prob:.2f})"
#                     color = (0, 0, 255)  # Red for unknown persons
#
#                 # Display result on frame with tight bounding box
#                 cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
#                 cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
#             except Exception as e:
#                 print(f"Error processing frame: {e}")
#
#     # Show the frame
#     cv2.imshow("Face Recognition", frame)
#
#     # Exit on 'q' press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()
# print("✅ Real-time testing completed!")


import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template_string
from insightface.app import FaceAnalysis
import pickle
import mediapipe as mp
import os

# Get absolute path to the current file directory (FaceRecognition folder)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "face_recognizer.pkl")

with open(MODEL_PATH, "rb") as f:
    svm = pickle.load(f)


# Initialize InsightFace (ArcFace) for embeddings
face_app = FaceAnalysis(providers=['CUDAExecutionProvider'])  # Use 'CUDAExecutionProvider' if GPU is available
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Initialize MediaPipe for face detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

# Confidence threshold for "Unknown" detection
CONFIDENCE_THRESHOLD = 0.7  # Adjust as needed


def preprocess_image(image):
    """Detect and crop faces using MediaPipe, resize for ArcFace embedding."""
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)
    if not results.detections:
        return []

    faces_data = []
    h, w, _ = image.shape

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box

        x_min = int(bbox.xmin * w)
        y_min = int(bbox.ymin * h)
        box_width = int(bbox.width * w)
        box_height = int(bbox.height * h)
        x_max = x_min + box_width
        y_max = y_min + box_height

        # Expand bounding box for better feature extraction
        expand_top, expand_side, expand_bottom = 1.0, 0.25, 0.2
        crop_x_min = max(0, int(x_min - expand_side * box_width))
        crop_y_min = max(0, int(y_min - expand_top * box_height))
        crop_x_max = min(w, int(x_min + (1 + 2 * expand_side) * box_width))
        crop_y_max = min(h, int(y_min + (1 + expand_top + expand_bottom) * box_height))

        face_crop = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        if face_crop.size == 0:
            continue

        face_resized = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)
        faces_data.append({
            'image': cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB),
            'bbox': (x_min, y_min, x_max, y_max)
        })

    return faces_data


def generate_frames():
    """Start IP camera stream and generate video frames with predictions."""
    ip_camera_url = 'http://192.168.18.69:8080/video'  # <-- Replace this with your phone IP camera URL
    cap = cv2.VideoCapture(ip_camera_url)
    if not cap.isOpened():
        print("Error: Could not open IP camera stream.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to get frame from IP camera.")
                break

            faces_data = preprocess_image(frame)

            for face_data in faces_data:
                preprocessed_img = face_data['image']
                bbox = face_data['bbox']
                try:
                    faces = face_app.get(preprocessed_img)
                    if not faces:
                        continue

                    embedding = faces[0].embedding  # 512D
                    embedding = embedding.reshape(1, -1)

                    pred_name = svm.predict(embedding)[0]
                    pred_prob = svm.predict_proba(embedding)[0].max()

                    if pred_prob >= CONFIDENCE_THRESHOLD:
                        label = f"{pred_name} ({pred_prob:.2f})"
                        color = (0, 255, 0)  # Green for known persons
                    else:
                        label = f"Unknown ({pred_prob:.2f})"
                        color = (0, 0, 255)  # Red for unknown persons

                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                except Exception as e:
                    print(f"Error processing frame: {e}")

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        cap.release()
        print("IP camera stream released.")


def analyze_face_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    frame_skip = 6  # Process every 6th frame
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            faces_data = preprocess_image(frame)

            for face_data in faces_data:
                preprocessed_img = face_data['image']
                try:
                    faces = face_app.get(preprocessed_img)
                    if not faces:
                        continue

                    embedding = faces[0].embedding
                    embedding = embedding.reshape(1, -1)

                    pred_name = svm.predict(embedding)[0]
                    pred_prob = svm.predict_proba(embedding)[0].max()

                    if pred_prob >= 0.85:  # Confidence threshold
                        cap.release()
                        return {
                            "name": pred_name,
                            "gender": "Male" if "male" in pred_name.lower() else "Female",  # crude gender estimation
                            "confidence": round(float(pred_prob), 2)
                        }

                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue

            frame_idx += 1

    finally:
        cap.release()

    return {"name": "Unknown", "gender": "Unknown", "confidence": 0.0}


# @app.route('/')
# def index():
#     """Serve the webpage with video feed."""
#     return render_template_string('''
#         <!DOCTYPE html>
#         <html>
#         <head>
#             <title>Face Recognition</title>
#             <style>
#                 body { font-family: Arial, sans-serif; text-align: center; margin-top: 20px; }
#                 h1 { color: #333; }
#                 img { max-width: 100%; height: auto; }
#             </style>
#         </head>
#         <body>
#             <h1>Real-Time Face Recognition</h1>
#             <img src="{{ url_for('video_feed') }}" alt="Video Feed">
#         </body>
#         </html>
#     ''')


# @app.route('/video_feed')
# def video_feed():
#     """Stream video feed with predictions."""
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)



# import cv2
# import numpy as np
# import base64
# from flask import Flask, request, render_template
# from flask_socketio import SocketIO
# from insightface.app import FaceAnalysis
# import pickle
# import mediapipe as mp
# import traceback
#
# # Initialize Flask app
# app = Flask(__name__)
# socketio = SocketIO(app, cors_allowed_origins="*")
#
# # Load SVM model
# print("Loading SVM model...")
# try:
#     with open("face_recognizer.pkl", "rb") as f:
#         svm = pickle.load(f)
#     print("SVM model loaded successfully.")
# except Exception as e:
#     print(f"Error loading SVM model: {e}")
#     raise
#
# # Initialize InsightFace (ArcFace) for embeddings
# print("Initializing InsightFace...")
# try:
#     face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
#     face_app.prepare(ctx_id=0, det_size=(320, 320))
#     print("InsightFace initialized.")
# except Exception as e:
#     print(f"Error initializing InsightFace: {e}")
#     raise
#
# # Initialize MediaPipe for face detection
# print("Initializing MediaPipe...")
# try:
#     mp_face_detection = mp.solutions.face_detection
#     face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.3)
#     print("MediaPipe initialized.")
# except Exception as e:
#     print(f"Error initializing MediaPipe: {e}")
#     raise
#
# # Confidence threshold for "Unknown" detection
# CONFIDENCE_THRESHOLD = 0.7
#
#
# def preprocess_image(image):
#     try:
#         rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = face_detector.process(rgb_frame)
#         if not results.detections:
#             return []
#
#         faces_data = []
#         h, w, _ = image.shape
#         for detection in results.detections:
#             bbox = detection.location_data.relative_bounding_box
#             x_min = int(bbox.xmin * w)
#             y_min = int(bbox.ymin * h)
#             box_width = int(bbox.width * w)
#             box_height = int(bbox.height * h)
#             x_max = x_min + box_width
#             y_max = y_min + box_height
#
#             face_crop = image[y_min:y_max, x_min:x_max]
#             if face_crop.size == 0:
#                 continue
#
#             face_resized = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)
#             faces_data.append(
#                 {'image': cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB), 'bbox': (x_min, y_min, x_max, y_max)})
#
#         return faces_data
#     except Exception as e:
#         print(f"Error in preprocess_image: {e}")
#         return []
#
#
# def process_frame(frame_data):
#     try:
#         img_data = base64.b64decode(frame_data.split(',')[1])
#         np_img = np.frombuffer(img_data, np.uint8)
#         frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
#         faces_data = preprocess_image(frame)
#
#         for face_data in faces_data:
#             preprocessed_img = face_data['image']
#             bbox = face_data['bbox']
#             faces = face_app.get(preprocessed_img)
#             if not faces:
#                 continue
#
#             embedding = faces[0].embedding.reshape(1, -1)
#             pred_name = svm.predict(embedding)[0]
#             pred_prob = svm.predict_proba(embedding)[0].max()
#             label = f"{pred_name} ({pred_prob:.2f})" if pred_prob >= CONFIDENCE_THRESHOLD else f"Unknown ({pred_prob:.2f})"
#             color = (0, 255, 0) if pred_prob >= CONFIDENCE_THRESHOLD else (0, 0, 255)
#
#             cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
#             cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
#
#         ret, buffer = cv2.imencode('.jpg', frame)
#         return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
#     except Exception as e:
#         print(f"Error processing frame: {traceback.format_exc()}")
#         return None
#
#
# @socketio.on('process_frame')
# def handle_frame(data):
#     processed_frame = process_frame(data['frame'])
#     if processed_frame:
#         socketio.emit('frame_response', {'processed_frame': processed_frame})
#     else:
#         socketio.emit('frame_response', {'error': 'Failed to process frame'})
#
#
# @socketio.on('connect')
# def handle_connect():
#     print("Client connected")
#
#
# @socketio.on('disconnect')
# def handle_disconnect():
#     print("Client disconnected")
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# if __name__ == "__main__":
#     print("Starting Flask server with SocketIO...")
#     socketio.run(app, host="0.0.0.0", port=5001, debug=False)
