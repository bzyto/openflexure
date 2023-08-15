# main.py
import sys
import os
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
"""
this code takes path to a folder as an input and writes filenames of images with cracks detected into a txt file in the same folder
it can/should be called from the terminal
NOTES for myself
make sure to change modelpath below
libraries versions
cv2 - 4.5.3
numpy - 1.21.6
tflite-runtime 2.11.0
all the rest should be installed
scan path - :/var/openflexure/application/openflexure-microscope-server/openflexure_microscope/api/default_extensions/scan.py
setup - /var/openflexure/application/openflexure-microscope-server/setup.py
"""
def tflite_detect_images(image_path):
    modelpath = '/home/pi/tensorflow_object_detection/hopethisworks.tflite'
    # Load the Tensorflow Lite model into memory
    interpreter = tflite.Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    #checking type of model. In future someone could convert the model into unit8 type, then inference could be faster.
    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5
    try:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except cv2.error:
        return False
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
    for i in range(len(scores)):
        if ((scores[i] > 0.2) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            area =abs((xmax-xmin)*(ymax-ymin))
            ## remove large false positives
            # we care for small cracks hence if a large crack is not detected it is fine
            if area >=0.4*(imH*imW):
                break
            #in case the detection is longer than taller we get rid of it because cracks are long and thin.
            if (xmax-xmin)>=(ymax-ymin):
                break
            #print(True)
            return True
    #print(False)
    return False
def write_detection_to_txt(folder_path):
    found_detections = False
    with open(os.path.join(folder_path, 'detections.txt'), 'w') as w:
        for image in os.listdir(folder_path):
            detection = tflite_detect_images(os.path.join(folder_path, image))
            if detection:
                found_detections = True
                w.write(image+'\n')
        if not found_detections:
            w.write("None")

#write_detection_to_txt("/home/pi/tensorflow_object_detection/main_test/borys/folder_test")
def main():
    folder_path = sys.argv[1]
    write_detection_to_txt(folder_path)
if __name__=="__main__":
    main()
"""
code to write in scan.py

import subprocess
bash_command = "python3 read_and_detect.py path_to_folder
# Execute the Bash command and capture the output and errors
completed_process = subprocess.run(bash_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Print the output and errors
#print("Command output:")
#print(completed_process.stdout)
# Check the return code
if completed_process.returncode == 0:
    print("Command executed successfully.")
else:
    print("Command failed with return code:", completed_process.returncode)


"""