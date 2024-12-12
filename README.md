# DotPad Meituan

## 📌Project Objective
The goal of the project is to develop applications that help with tactile graphic learning.  
To this end, the Meituan team's proposal is to convert images into text and then explain the text by voice to help visually impaired people recognize household items.

---

## Technical Libraries

### 1. Image Recognition
- **YOLOv8 Model**: Used for image detection.  
- **OpenCV**: Crops the object regions detected by YOLO, performs scaling, contrast enhancement, edge detection, and final image processing on grayscale images.  
- **NumPy**: Calculates the area of each test box and maps the image detection information onto a graphical background.  

Using the above technologies, the system integrates the processes to display the cropped image, grayscale image, and edge-detected image on a webpage. On the backend, it generates an array for transmission to the `dotpad` for visualization.

### 2. Text Generation
- **Encoder**: Utilizes ResNet101.  
- **Decoder**: Uses an attention-based RNN decoder.  
- **Dataset**: The model is trained on the COCO2014 dataset to generate a text generation model.

### 3. Web Development
- **Django**:
  - **Model**: Interacts with the database, defines data structures, and handles storage logic.  
  - **View**: Processes business logic, connects models and templates, and manages data transfer and processing.  
  - **Template**: Renders the user interface and displays data on HTML pages.  
  - **Controller**: Implements routing through `urls.py`, assigning requests to the appropriate view functions.  

The system processes the original image through the image detection model, generates corresponding text based on the processed image, and uses Google Chrome's built-in read-aloud functionality to read the text. Additionally, it provides a button to send the image to the `dotpad` for further display.

## 🛠️ Team Contributions

| Member        | Contributions                                                                                                                  | Percentage        |
|---------------|-------------------------------------------------------------------------------------------------------------------------------|-------------------|
| **진수림**       | - Web Development <br> - Frontend Code <br> - Backend Code <br> - Frontend-Backend Integration <br> - Text-to-Speech Feature <br> - Text Generation Model <br> - Final PPT Revision    <br> - DotPad Display Functionality                               | **50%**           |
| **수림봉**       | - Image Display <br> - Image Recognition Model <br> - Text Generation Model <br> - DotPad Display Functionality <br> - DotPad Integration (Frontend-Backend Link) <br> - Final Presentation Script Revision <br> - Technical Documentation Writing <br> - Report Writing                                    | **40%**           |
| **양신뢰**       | - PPT Creation and Presentation Script <br> - Frontend Code                                                                            | **5%**            |
| **장요탁**       | - PPT Creation and Presentation Script <br> - Frontend Code                                                                            | **5%**            |


PS. The contribution percentages for this project were evaluated by the team leader (**진수림**) based on actual participation. In this project, **진수림** made the greatest contributions, with **수림봉** providing substantial assistance. The other two team members had minimal involvement in the project development, making only minor adjustments to the layout of a single screen in the frontend.
Additionally, the PPT and presentation drafts created by the other two members were of low quality, with logical inconsistencies, incomplete content, and numerous AI-generated sections. These required secondary revisions by the team leader and **수림봉** before they could be finalized for submission.
This assessment is based solely on the team leader's perspective. If there are any concerns regarding the contribution evaluation, please contact the team leader for further clarification.

---

## 📚 Features
1. **Image-to-Text Conversion**: Converts input images into corresponding text descriptions.
2. **Text-to-Speech**: Reads out the generated text to help visually impaired users understand.
3. **DotPad Display**: Supports tactile graphic display on DotPad devices.
4. **Integration**: Seamlessly links the frontend, backend, and DotPad device for smooth operation.



---


## 🚀 How to Run

### Step 1: Download the Model Files
You need to download the required model files from the following link:
[Google Drive - Model Files](https://drive.google.com/drive/folders/1ZyE13mffdJcaNbRVeaDR_xg42sDUG-MU?usp=sharing)

### Step 2: Clone the Repository and Start the Server
Run the following commands in your terminal:
```bash
git clone https://github.com/slplsp/dotpad_meituan.git
cd dotpad_meituan
python manage.py runserver

   
---


## 📞 Contact

If you have any questions, feel free to reach out via email:  
- suilinpeng15@gmail.com
- sl695969@outlook.com




