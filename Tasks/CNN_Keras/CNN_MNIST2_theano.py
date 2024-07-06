import numpy as np
import argparse
import cv2
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD  # Keras with Theano backend
from keras.utils import to_categorical
from cnn.neural_network2 import CNN  # Importing the CNN class

# Set Keras backend to Theano
import os
os.environ['KERAS_BACKEND'] = 'theano'

# Parse the Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save_model", type=int, default=-1)
ap.add_argument("-l", "--load_model", type=int, default=-1)
ap.add_argument("-w", "--save_weights", type=str)
ap.add_argument("-o", "--output_dir", type=str, default="output")
args = vars(ap.parse_args())

# Read/Download MNIST Dataset
print('Loading MNIST Dataset...')
dataset = fetch_openml('mnist_784')

# Convert DataFrame to numpy array and reshape
mnist_data = dataset.data.to_numpy().reshape((dataset.data.shape[0], 28, 28, 1))
mnist_data = mnist_data.astype('float32') / 255.0

# Divide data into testing and training sets
train_img, test_img, train_labels, test_labels = train_test_split(
    mnist_data,
    dataset.target.astype("int"),
    test_size=0.1
)

# Transform training and testing labels to one-hot encoding
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Initialize the CNN model from your custom module
print('\nCompiling CNN model...')
clf = CNN.build(width=28, height=28, depth=1, total_classes=10, Saved_Weights_Path=args["save_weights"] if args["load_model"] > 0 else None)

# Use SGD optimizer without decay parameter
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# If not loading a pre-trained model, train the model
if args["load_model"] < 0:
    print('\nTraining the Model...')
    clf.fit(train_img, train_labels, batch_size=128, epochs=20, verbose=1)
    
    # Evaluate accuracy and loss on test data
    print('\nEvaluating Accuracy and Loss...')
    loss, accuracy = clf.evaluate(test_img, test_labels, batch_size=128, verbose=1)
    print('Accuracy of Model: {:.2f}%'.format(accuracy * 100))

# Save the trained model if specified
if args["save_model"] > 0:
    print('Saving model to file...')
    # Ensure the filename ends with .weights.h5
    if not args["save_weights"].endswith(".weights.h5"):
        args["save_weights"] += ".weights.h5"
    clf.save_weights(args["save_weights"], overwrite=True)

# Ensure the output directory exists
if not os.path.exists(args["output_dir"]):
    os.makedirs(args["output_dir"])

# Show predictions and images for random test samples
print('\nShowing Predictions:')
for num in np.random.choice(np.arange(0, len(test_labels)), size=(5,)):
    # Predict the label of digit using CNN.
    probs = clf.predict(test_img[num][np.newaxis, ...])
    prediction = np.argmax(probs)

    # Display the image
    image = (test_img[num] * 255).astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for OpenCV display
    image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, f'Pred: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    actual_value = np.argmax(test_labels[num])
    cv2.putText(image, f'Actual: {actual_value}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Save the image to a file
    output_path = os.path.join(args["output_dir"], f"sample_{num}.png")
    cv2.imwrite(output_path, image)

    # Show the image
    cv2.imshow('Digit and Prediction', image)
    cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
