# Privacy Policy — EduBridge Mini Project 8

**Project Name:** NeuralVision AI / EduBridge Mini Project 8 (Handwritten Digit Recognition)  
**Developer & Maintainer:** Harish  
**Contact Email:** [harishhari781823@gmail.com](mailto:harishhari781823@gmail.com)  
**Repository:** [https://github.com/Haris87-star/edubridge-miniproject8](https://github.com/Haris87-star/edubridge-miniproject8)  
**Effective Date:** October 28, 2026  

---

## 1. Overview & Purpose
This Privacy Policy outlines how the **EduBridge Mini Project 8 (Handwritten Digit Recognition System)** collects, processes, stores, and protects user data. As an educational and open-source machine learning project, user privacy, data security, and transparent processing are core design principles.

## 2. Information Collected & Processed
- **Digit Drawings & Uploaded Images:** When you draw digits on the interactive HTML5 canvas or upload an image file (PNG, JPG, JPEG, BMP), the image is loaded into volatile memory solely for preprocessing (resizing, polarity detection, center-of-mass centering) and immediate model inference.
- **No Personal Identifiable Information (PII):** The application does not collect, track, or record any personally identifiable information such as names, IP addresses, location data, phone numbers, or user credentials.
- **No External Cloud Storage:** Uploaded or drawn images are strictly processed locally and are **never** uploaded to third-party cloud servers, tracking APIs, analytics networks, or external databases.

## 3. Data Storage & Retention
- **In-Memory Volatile Processing:** All image processing and neural network predictions occur dynamically within local memory during your active Streamlit runtime session.
- **Session Reset:** Once the user clears the canvas, navigates away, or stops the application session, temporary array buffers in memory are automatically cleared and discarded.
- **Offline Compatibility:** The entire machine learning inference engine executes offline on CPU without requiring an internet connection once the local environment dependencies are installed.

## 4. Third-Party Services & Dependencies
This application relies on trusted open-source Python libraries (TensorFlow, NumPy, Pandas, Matplotlib, Streamlit, and Pillow). None of these packages are configured to exfiltrate user data or collect telemetry for advertising or commercial monetization.

## 5. Security Measures
- **Input Validation:** Strict file type validation and size limits (max 5 MB) prevent memory overload and arbitrary payload execution.
- **Safe Evaluation:** Model evaluation uses pre-packaged benchmark datasets (MNIST) stored in local read-only format without executing untrusted remote code.

## 6. Academic & Open-Source Integrity
This project is released under the **MIT License** for academic, research, and non-commercial educational review as part of the **EduBridge Machine Learning Curriculum (Mini Project 8)**.

## 7. Contact & Queries
If you have any questions, feedback, or concerns regarding this Privacy Policy or the data handling practices of this project, please contact:

- **Developer:** Harish  
- **Email:** [harishhari781823@gmail.com](mailto:harishhari781823@gmail.com)  
- **GitHub:** [https://github.com/Haris87-star](https://github.com/Haris87-star)  
