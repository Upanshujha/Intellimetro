# Intellimetro: Smart Monitoring System
IntelliMetro is an innovative surveillance solution designed to enhance metro coach monitoring through advanced computer vision technologies. The system performs real-time occupancy analysis and person of interest identification, making metro travel safer and more efficient.

🚀 Features
Real-Time Occupancy Monitoring

Detects the number of people in a metro coach using YOLOv3 object detection.
Calculates and displays occupancy percentage based on the defined coach capacity.
Highlights overcrowding situations with actionable alerts.
Face Recognition for Security

Identifies persons of interest by comparing faces in the video feed against a preloaded database.
Alerts when a recognized individual is detected.
Interactive User Interface

User-friendly Streamlit-based application.
Allows dynamic updates to the criminal database by adding new persons of interest (photo and name).
Provides real-time metrics, security alerts, and visual overlays on video frames.
Video Analysis

Supports video uploads in formats like MP4, AVI, and MOV.
Processes frames efficiently for faster analysis while maintaining accuracy.
Generates a summary report on occupancy and security incidents.
🛠 Technologies Used
Programming Language: Python
Libraries: OpenCV, face_recognition, Streamlit, NumPy
Model: YOLOv3 (You Only Look Once) for object detection
📂 Project Structure
app.py: Main application codebase for IntelliMetro.
criminals/: Folder containing images of persons of interest.
yolo-coco/: Directory holding YOLOv3 configuration and weight files.
🎯 How to Use
Clone the repository and set up the environment.

bash
Copy code
git clone <repository-link>
cd IntelliMetro
pip install -r requirements.txt
Add the YOLOv3 configuration and weight files in the yolo-coco/ directory.

Run the application:

bash
Copy code
streamlit run app.py
Use the sidebar to:

Upload metro coach videos for analysis.
Add new persons of interest to the criminal database.
Configure coach capacity settings.
View real-time analysis and receive alerts for overcrowding and detected persons of interest.

📈 Future Enhancements
Incorporate additional object detection models for enhanced accuracy.
Extend functionality for live video stream analysis.
Add features for crowd behavior prediction and emergency management.
🙌 Contributions
Contributions are welcome! Feel free to open issues or submit pull requests for new features or improvements.

📝 License
This project is licensed under the MIT License. See the LICENSE file for details.
