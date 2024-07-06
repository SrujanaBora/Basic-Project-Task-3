import sys
import numpy as np
import pickle
import sys
from sklearn import model_selection, svm, preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from MNIST_Dataset_Loader.mnist_loader import MNIST
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# Save all the print statements in a log file
old_stdout = sys.stdout
log_file = open("summary.log", "w")
sys.stdout = log_file

# Load MNIST data
print('\nLoading MNIST Data...')
data = MNIST('./MNIST_Dataset_Loader/dataset/')

print('\nLoading Training Data...')
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)

print('\nLoading Testing Data...')
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)

# Features and labels
X = train_img
y = train_labels

# Prepare classifier training and testing data
print('\nPreparing Classifier Training and Validation Data...')
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

# Train the SVM classifier
print('\nSVM Classifier with gamma = 0.1; Kernel = polynomial')
clf = svm.SVC(gamma=0.1, kernel='poly')
clf.fit(X_train, y_train)

# Pickle the classifier for future use
print('\nPickling the Classifier for Future Use...')
with open('MNIST_SVM.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('MNIST_SVM.pickle', 'rb')
clf = pickle.load(pickle_in)

# Calculate the accuracy of the trained classifier
print('\nCalculating Accuracy of trained Classifier...')
acc = clf.score(X_test, y_test)
print('\nSVM Trained Classifier Accuracy:', acc)

# Make predictions on validation data
print('\nMaking Predictions on Validation Data...')
y_pred = clf.predict(X_test)

# Calculate the accuracy of predictions
print('\nCalculating Accuracy of Predictions...')
accuracy = accuracy_score(y_test, y_pred)
print('\nAccuracy of Classifier on Validation Images:', accuracy)
print('\nPredicted Values: ',y_pred)

# Create confusion matrix
print('\nCreating Confusion Matrix...')
conf_mat = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:\n', conf_mat)

# Plot confusion matrix for validation data
plt.matshow(conf_mat)
plt.title('Confusion Matrix for Validation Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix_validation.png')
plt.close()

# Make predictions on test input images
print('\nMaking Predictions on Test Input Images...')
test_labels_pred = clf.predict(test_img)

# Calculate accuracy of the trained classifier on test data
print('\nCalculating Accuracy of Trained Classifier on Test Data...')
acc = accuracy_score(test_labels, test_labels_pred)
print('\nAccuracy of Classifier on Test Images:', acc)

# Create confusion matrix for test data
print('\nCreating Confusion Matrix for Test Data...')
conf_mat_test = confusion_matrix(test_labels, test_labels_pred)
print('\nPredicted Labels for Test Images: ',test_labels_pred)
print('\nConfusion Matrix for Test Data:\n', conf_mat_test)

# Plot confusion matrix for test data
plt.matshow(conf_mat_test)
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.savefig('confusion_matrix_test.png')
plt.close()

# Restore the original stdout
sys.stdout = old_stdout
log_file.close()

# Show the test images with original and predicted labels
a = np.random.randint(1, len(test_img), 15)
for i in a:
    two_d = (np.reshape(test_img[i], (28, 28)) * 255).astype(np.uint8)
    plt.title('Original Label: {0}  Predicted Label: {1}'.format(test_labels[i], test_labels_pred[i]))
    plt.imshow(two_d, interpolation='nearest', cmap='gray')
    plt.savefig(f'test_image_{i}.png')
    plt.close()

