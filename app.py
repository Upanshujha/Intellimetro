import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import face_recognition
from datetime import datetime



class FaceRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_criminal_faces()
        
    def load_criminal_faces(self):
        criminals_dir = "C:/Users/rbham/Desktop/IntelliSense/criminals"
        if not os.path.exists(criminals_dir):
            os.makedirs(criminals_dir)
            
        for filename in os.listdir(criminals_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                try:
                    # Load criminal image
                    image_path = os.path.join(criminals_dir, filename)
                    face_image = face_recognition.load_image_file(image_path)
                    
                    # Get face encoding
                    face_encoding = face_recognition.face_encodings(face_image)[0]
                    
                    # Store encoding and name (filename without extension)
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(os.path.splitext(filename)[0])
                except Exception as e:
                    st.warning(f"Failed to load criminal image {filename}: {str(e)}")

                    
    def identify_faces(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            # Compare with known criminal faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding,
                tolerance=0.6
            )
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
                
            face_names.append(name)
            
        return face_locations, face_names

class PersonDetector:
    def __init__(self):
        # Load YOLO
        self.yolo_dir = "C:/Users/rbham/Desktop/IntelliSense/yolo-coco"  # Directory containing YOLO files
        weights_path = os.path.join(self.yolo_dir, "yolov3.weights")
        config_path = os.path.join(self.yolo_dir, "yolov3.cfg")
        
        # Check if files exist
        if not os.path.exists(weights_path) or not os.path.exists(config_path):
            st.error("YOLO files not found! Please ensure yolov3.weights and yolov3.cfg are in the yolo-coco directory.")
            st.stop()
            
        # Load YOLO network
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # Get output layer names
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
    def detect_persons(self, frame):
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Forward pass through network
        outputs = self.net.forward(self.output_layers)
        
        # Initialize lists for detected boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each output
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for person class (class ID 0 in COCO dataset)
                if class_id == 0 and confidence > 0.5:
                    # Get box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maxima suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        return len(indexes), boxes, indexes, confidences
        
    def calculate_occupancy(self, count, capacity=300):
        return (count / capacity) * 100


class IntelliMetroApp:
    def __init__(self):
        self.detector = PersonDetector()
        self.face_recognizer = FaceRecognizer()
        
    def add_criminal(self):
        st.sidebar.header("Add Person of Interest")
        
        # Store the uploaded file but don't process it yet
        uploaded_file = st.sidebar.file_uploader(
            "Upload Criminal Photo", 
            type=['jpg', 'jpeg', 'png']
        )
        
        # Get criminal name
        criminal_name = st.sidebar.text_input("Criminal Name")
        
        # Only show the Add button if both photo and name are provided
        if uploaded_file is not None and criminal_name:
            if st.sidebar.button("Add Criminal"):
                try:
                    # Create criminals directory if it doesn't exist
                    if not os.path.exists("C:/Users/rbham/Desktop/IntelliSense/criminals"):
                        os.makedirs("C:/Users/rbham/Desktop/IntelliSense/criminals")
                        
                    # Save uploaded image
                    save_path = f"C:/Users/rbham/Desktop/IntelliSense/criminals/{criminal_name}.jpg"
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    # Reload face recognizer
                    self.face_recognizer = FaceRecognizer()
                    st.sidebar.success(f"Successfully added {criminal_name} to database!")
                    
                    # Force reload of the entire app to refresh the face recognizer
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.sidebar.error(f"Error adding criminal: {str(e)}")
        elif uploaded_file is not None or criminal_name:
            st.sidebar.info("Please provide both photo and name to add a person of interest.")


    def add_criminal(self):
        st.sidebar.header("Add Person of Interest")
        
        # Initialize session state for video if not exists
        if 'video_data' not in st.session_state:
            st.session_state.video_data = None
            
        # Initialize session state for processing flag if not exists
        if 'is_processing' not in st.session_state:
            st.session_state.is_processing = False
        
        # Store the uploaded file but don't process it yet
        uploaded_file = st.sidebar.file_uploader(
            "Upload Criminal Photo", 
            type=['jpg', 'jpeg', 'png'],
            key="criminal_upload"  # Add unique key
        )
        
        # Get criminal name
        criminal_name = st.sidebar.text_input("Criminal Name", key="criminal_name")  # Add unique key
        
        # Only show the Add button if both photo and name are provided
        if uploaded_file is not None and criminal_name:
            if st.sidebar.button("Add Criminal"):
                try:
                    # Create criminals directory if it doesn't exist
                    if not os.path.exists("C:/Users/rbham/Desktop/IntelliSense/criminals"):
                        os.makedirs("C:/Users/rbham/Desktop/IntelliSense/criminals")
                        
                    # Save uploaded image
                    save_path = f"C:/Users/rbham/Desktop/IntelliSense/criminals/{criminal_name}.jpg"
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    # Reload face recognizer
                    self.face_recognizer = FaceRecognizer()
                    st.sidebar.success(f"Successfully added {criminal_name} to database!")
                    
                    # Only rerun if we have a video being processed
                    if st.session_state.video_data is not None:
                        st.experimental_rerun()
                    
                except Exception as e:
                    st.sidebar.error(f"Error adding criminal: {str(e)}")
        elif uploaded_file is not None or criminal_name:
            st.sidebar.info("Please provide both photo and name to add a person of interest.")




    
    def process_video(self, video_path, coach_capacity):
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Add frame skip selector
        frame_skip = 15
        
        st.subheader("Video Analysis")
        progress_bar = st.progress(0)
        
        # Create columns for layout
        col1, col2 = st.columns([1, 2])
        frame_placeholder = col1.empty()
        
        # Create metrics columns
        metrics_col1, metrics_col2, metrics_col3 = col2.columns(3)
        metrics = {
            'current': metrics_col1.empty(),
            'max': metrics_col2.empty(),
            'min': metrics_col3.empty()
        }
        
        # Create alert boxes
        status_placeholder = col2.empty()
        criminal_alert_placeholder = col2.empty()
        
        frame_count = 0
        max_occupancy = 0
        min_occupancy = 100
        detected_criminals = set()  # Track unique criminals detected
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip == 0:
                # Detect persons and calculate occupancy
                count, boxes, indexes, confidences = self.detector.detect_persons(frame)
                occupancy = self.detector.calculate_occupancy(count, coach_capacity)
                
                # Update occupancy stats
                max_occupancy = max(max_occupancy, occupancy)
                min_occupancy = min(min_occupancy, occupancy)
                
                # Perform face recognition
                face_locations, face_names = self.face_recognizer.identify_faces(frame)
                
                # Draw person detection boxes
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw face detection boxes and labels
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    if name != "Unknown":
                        # Add to detected criminals set
                        detected_criminals.add(name)
                        # Draw red box for criminals
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, name, (left, top - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Add occupancy information
                cv2.putText(frame, f"Persons: {count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Occupancy: {occupancy:.1f}%", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Resize and display frame
                display_width = 400
                aspect_ratio = frame.shape[1] / frame.shape[0]
                display_height = int(display_width / aspect_ratio)
                display_frame = cv2.resize(frame, (display_width, display_height))
                frame_placeholder.image(display_frame, channels="BGR")
                
                # Update metrics
                metrics['current'].metric(
                    "Current Occupancy", 
                    f"{occupancy:.1f}%",
                    f"{occupancy - min_occupancy:+.1f}%"
                )
                metrics['max'].metric("Maximum", f"{max_occupancy:.1f}%")
                metrics['min'].metric("Minimum", f"{min_occupancy:.1f}%")
                
                # Update occupancy status
                if occupancy > 90:
                    status_placeholder.error("⚠️ OVERCROWDED: Coach is extremely full!")
                elif occupancy > 75:
                    status_placeholder.warning("⚡ HIGH OCCUPANCY: Coach is getting crowded")
                elif occupancy > 50:
                    status_placeholder.info("✅ MODERATE OCCUPANCY: Coach is comfortably occupied")
                else:
                    status_placeholder.success("🆓 LOW OCCUPANCY: Coach has plenty of space")
                
                # Show criminal alerts
                if detected_criminals:
                    criminal_alert_placeholder.error(
                        f"🚨 ALERT: Detected persons of interest: {', '.join(detected_criminals)}"
                    )
            
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
        
        cap.release()
        
        # Display final summary
        st.subheader("Analysis Summary")
        col1, col2 = st.columns(2)
        
        # Occupancy summary
        col1.write("### Occupancy Statistics")
        col1.write(f"""
        - Total Frames: {total_frames}
        - Frames Analyzed: {total_frames // frame_skip}
        - Video Duration: {total_frames/fps:.2f} seconds
        - Maximum Occupancy: {max_occupancy:.1f}%
        - Minimum Occupancy: {min_occupancy:.1f}%
        """)
        
        # Security summary
        col2.write("### Security Report")
        if detected_criminals:
            col2.error(f"""
            Detected {len(detected_criminals)} person(s) of interest:
            {', '.join(detected_criminals)}
            """)
        else:
            col2.success("No persons of interest detected")
        
        st.success("Video analysis completed successfully!")
    
    def main(self):
        st.set_page_config(
            page_title="IntelliMetro",
            page_icon="🚇",
            layout="wide"
        )
        
        st.title("🚇 IntelliMetro Smart Monitoring System")
        
        # Initialize session state for video if not exists
        if 'video_data' not in st.session_state:
            st.session_state.video_data = None
        
        # Sidebar
        st.sidebar.title("Controls")
        
        # Add criminal database management
        self.add_criminal()
        
        # Coach capacity input
        coach_capacity = st.sidebar.number_input(
            "Enter Coach Capacity",
            min_value=1,
            max_value=1000,
            value=300
        )
        
        # Video upload
        uploaded_file = st.file_uploader(
            "Upload Metro Coach Video",
            type=['mp4', 'avi', 'mov'],
            key="video_upload"
        )
        
        # Check if video was removed
        if uploaded_file is None:
            st.session_state.video_data = None
            st.empty()  # Clear the UI
            return  # Exit the function to stop processing
        
        # If video is uploaded
        if uploaded_file is not None:
            # Store video data in session state
            st.session_state.video_data = uploaded_file.read()
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(st.session_state.video_data)
            self.process_video(tfile.name, coach_capacity)


if __name__ == "__main__":
    app = IntelliMetroApp()
    app.main()

