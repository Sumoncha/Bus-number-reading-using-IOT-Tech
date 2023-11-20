import torch
import numpy as np
import cv2
from PIL import Image
import IPython.display as ipd
import easyocr
import os
import gtts
reader = easyocr.Reader(lang_list=['en'])

def imageData(filtered_result, roi):

    "Function that takes the result from easyocr and frame, displays it, and converts text to speech"
    towrite = []
    for (bbox, text, prob) in filtered_result:
        if prob >= 0.91:
            # Displays text and probability of
            print(f'Detected text: {text} (Probability: {prob:.2f})')
            # Gets top-left and bottom-right bbox vertices
            (top_left, top_right, bottom_right, bottom_left) = bbox

            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

            # Creates a rectangle for bbox display
            cv2.rectangle(roi, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=2)

            # put recognized text
            cv2.putText(roi, text=text, org=(top_left[0] + 5, top_left[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.9, color=(0, 0, 0), thickness=4)

            # Appends text in a list to convert it to speech later
            towrite.append(text)
            roi_filename = f'detected_roi_{text}.jpg'
            
    
            cv2.imwrite(roi_filename, roi)
            
            
    # Convert text in the frame to speech
    if towrite != []:
        cv2.imshow("frame", roi)
        txtToSpeech(towrite,filtered_result)

def txtToSpeech(text,filtered_result):
    global language
    speech = ''
    digit_mapping = {
        '0': 'ศูนย์',
        '1': 'หนึ่ง',
        '2': 'สอง',
        '3': 'สาม',
        '4': 'สี่',
        '5': 'ห้า',
        '6': 'หก',
        '7': 'เจ็ด',
        '8': 'แปด',
        '9': 'เก้า'
    }
    for n in range(len(text)):
        speech = speech + ' ' + text[n]
    for text_result in filtered_result:
        digit = text_result[1]
        digit_text = ' '.join([digit_mapping[char] for char in digit])
        sound = gtts.gTTS('รถเมล์สาย'+digit_text+'มาแล้วค่ะ', lang='th')

        # Saving the converted audio in a mp3 file
        sound.save("text.mp3")

        # Playing the converted file
        os.system("text.mp3")
        length = len(speech)
        cv2.waitKey(length*150)
language = 'en'


def detect_and_crop_region_of_interest_from_camera(model_path):
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    
     
    # Open a connection to the camera (0 is typically the default camera)
    cap = cv2.VideoCapture(0)  # 0 for default camera, change to another number if you have multiple cameras
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        # Inference
        #imgHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = model(frame)
        results2 = model(frame)
        output_frame = results2.render()[0]
        resized_frame = cv2.resize(frame, (800, 600))
        gray_frame2 = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        bounding_box = results.pandas().xyxy[0]  # Get image predictions (pandas DataFrame)
        edges = cv2.Canny(gray_frame,100,200)
        blurred = cv2.GaussianBlur(edges,(5,5),0)
        thresh=cv2.threshold(blurred,80,255,cv2.THRESH_BINARY)[1]
        # Extract coordinates
        if not bounding_box.empty:
            x_min = int(bounding_box['xmin'][0])
            x_max = int(bounding_box['xmax'][0])
            y_min = int(bounding_box['ymin'][0])
            y_max = int(bounding_box['ymax'][0])
        
            # Crop the region of interest
            roi = gray_frame[y_min:y_max, x_min:x_max]
            
            # Perform OCR on the cropped region of interest using EasyOCR
            results = reader.readtext(roi, allowlist='0123456789')
            filtered_result = [item for item in results if item[2] > 0.91]
            imageData(filtered_result, roi)


            # Display the frame with detected text
            cv2.imshow('Frame with Text', roi)
        
        # Display the original frame from the camera
        cv2.imshow('Original Frame', gray_frame)
        
        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
model_path = 'weights/best.pt'
detect_and_crop_region_of_interest_from_camera(model_path)
