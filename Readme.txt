                                         HAND WRITTEN DIGITS RECOGNITION SYSTEM

1.)Start by executing the process_mnist.py file.

This script is responsible for preprocessing the MNIST dataset, which includes steps like normalizing the data, reshaping the images, and splitting the dataset into training and testing sets. CNN.h5 will be created in your folder after running.

2.)Run the single_digit_recognition.py File

After preprocessing is complete, run the single_digit_recognition.py script.
This file handles the recognition of individual handwritten digits using the trained model.
You can test the system by providing a single handwritten digit image and observing the predicted result.

3.)Run the multiple_digit_recognition.py File

Next, execute the multiple_digit_recognition.py script.
This script is designed for recognizing sequences of handwritten digits (e.g., numbers like 123 or 456).
Provide an input containing multiple digits to test the model's ability to correctly identify each digit in the sequence.

4.)Compare KNN vs. Random Forest Algorithms

Analyze the performance of two machine learning algorithms, K-Nearest Neighbors (KNN) and Random Forest.
Run the comparative study to observe differences in accuracy, computation time, and robustness for recognizing handwritten digits.
This step helps identify the most suitable algorithm for this recognition task.

5.)Examine the Training Accuracy vs. Validation Accuracy Curve

Visualize and analyze the training and validation accuracy curves.
This step involves generating a plot to observe how well the model is learning and generalizing.
Look for signs of overfitting or underfitting and use the insights to fine-tune the model if necessary.