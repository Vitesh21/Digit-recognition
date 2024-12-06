import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('CNN.h5')

# Function to get the predicted number and its confidence
def get_numbers(y_pred):
    number = np.argmax(y_pred)
    confidence = round(np.max(y_pred) * 100, 2)
    return str(number), confidence

# Open video capture
video = cv2.VideoCapture(0)

if video.isOpened():
    while True:
        # Read a frame from the video
        check, img = video.read()
        img2 = img.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gau = cv2.GaussianBlur(img_gray, (5, 5), 0)
        _, thresh = cv2.threshold(img_gau, 80, 255, cv2.THRESH_BINARY_INV)

        # Dilation to enhance the digits
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(thresh, kernel, iterations=1)

        # Edge detection
        edged = cv2.Canny(dilation, 50, 250)

        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through each contour and process
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Minimum area to filter noise
                x, y, w, h = cv2.boundingRect(contour)

                # Draw rectangle around the contour
                cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 255), 2)

                # Extract the digit region, preprocess, and resize to 28x28
                digit_img = thresh[y:y + h, x:x + w]
                digit_img_resized = cv2.resize(digit_img, (28, 28))
                im2arr = np.array(digit_img_resized).reshape(1, 28, 28, 1)
                im2arr = im2arr / 255.0  # Normalize the image

                # Predict the digit
                y_pred = model.predict(im2arr)
                num, confidence = get_numbers(y_pred)

                # Draw prediction text above the rectangle
                cv2.putText(img2, f'{num}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Display the results
        cv2.imshow("Frame", img2)
        cv2.imshow("Contours Frame", thresh)

        # Exit condition
        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to quit
            break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()