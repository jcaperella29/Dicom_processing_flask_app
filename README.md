 DICOM Flask App – Multi-Tab Lesion Detector
📌 Overview
This Flask web app allows users to upload DICOM files, detect lesions using DeepLesion (Faster R-CNN), and classify them with ResNet50. The UI features a multi-tab layout with a sidebar for smooth navigation.

✅ Upload & Process DICOM Files
✅ Detect & Classify Lesions Automatically
✅ Multi-Tab UI with Sidebar Navigation
✅ Sample DICOM File Included for Testing

🛠 Installation & Setup
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/YOUR_GITHUB_USERNAME/DICOM-Flask-App.git
cd DICOM-Flask-App
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
(If requirements.txt is missing, install manually:)

bash
Copy
Edit
pip install flask torch torchvision pydicom numpy matplotlib opencv-python
3️⃣ Run the Flask App
bash
Copy
Edit
python app.py
Then, open http://127.0.0.1:5000/ in your browser.

📂 Project Structure
graphql
Copy
Edit
DICOM-Flask-App/
│── static/                  # CSS & processed images
│   ├── style.css            # UI Styling (Velvet Room Theme)
│── templates/               # HTML Templates for Flask
│   ├── index.html           # Main UI (Tabs: Upload, Results)
│── uploads/                 # Stores uploaded DICOM files
│── sample.dcm               # Sample DICOM file for testing ✅
│── app.py                   # Flask Application
│── requirements.txt         # Dependencies
│── README.md                # This documentation
🚀 How to Use
1️⃣ Upload a DICOM File
Go to http://127.0.0.1:5000/
Click "Upload DICOM", select a file, and click "Upload & Process"
2️⃣ View Results
Click the "Results" tab to see detected lesions.
Lesions are shown with bounding boxes and classified as Tumor, Cyst, Hemorrhage, or Inflammation.
📝 Sample DICOM File
A sample DICOM file (se.dcm) is included in the repo for convenience.

If you don’t have a DICOM file, use this one for testing.
🔥 Future Upgrades
🔹 Grad-CAM Heatmaps – Highlight lesion focus areas.
🔹 DICOM Export – Save processed images back into DICOM format.
🔹 Automatic Report Generation – AI-generated text reports for findings.
