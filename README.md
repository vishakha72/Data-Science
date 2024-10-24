# Data-Science

Here’s a basic example of code in TensorFlow that creates and trains a simple neural network to classify handwritten digits from the popular MNIST dataset:
Key Components:
Data: MNIST dataset (60,000 training images, 10,000 test images).
Model: A simple Sequential neural network.
Flatten: Converts 28x28 pixel images into 1D vectors.
Dense: Fully connected layers with ReLU and softmax activations.
Dropout: Helps prevent overfitting.
Compilation: The model is compiled with Adam optimizer and sparse categorical crossentropy loss.
Training: The model is trained for 5 epochs.
Evaluation: The accuracy of the model is evaluated on the test dataset.
This code gives you a basic neural network for classifying images using TensorFlow.

1. Import Required Libraries
We import TensorFlow, which is the main library for building and training machine learning models.
layers, models: These are submodules of Keras (which is integrated into TensorFlow). layers helps to define neural network layers, and models helps to create and manage neural network models.
numpy: A numerical library used for data manipulation, though it's not explicitly used in this small code.

2. Load the MNIST Dataset
tf.keras.datasets.mnist: TensorFlow provides easy access to standard datasets like MNIST. This dataset contains 70,000 grayscale images of handwritten digits (28x28 pixels).
x_train, y_train: x_train is the training data (images), and y_train is the corresponding labels (digits 0-9).
x_test, y_test: These are the test data and labels used for evaluation.
load_data(): Loads the dataset and splits it into training and test sets (60,000 images for training, 10,000 for testing).

3. Normalize the Data
The pixel values in the MNIST images are between 0 and 255 (grayscale). To make learning easier, we normalize these values to be between 0 and 1 by dividing by 255.
This helps the model converge faster and perform better, as normalized inputs tend to improve gradient-based optimization.

4. Build the Model
We build a Sequential model, which means the layers are stacked in sequence, one after the other:
Flatten(input_shape=(28, 28)): Converts each 28x28 pixel image (a 2D matrix) into a 1D vector of 784 values. This is needed before passing the data to the fully connected (Dense) layer.
Dense(128, activation='relu'):

A Dense layer is a fully connected layer where each input is connected to each output.
This layer has 128 neurons, and the activation function used is ReLU (Rectified Linear Unit), which introduces non-linearity. ReLU replaces negative values with 0 and keeps positive values unchanged. It helps the network learn complex patterns.
Dropout(0.2): Dropout is a regularization technique used to prevent overfitting. Here, 20% of the neurons are randomly "dropped" (ignored) during training. This forces the model to generalize better by preventing it from relying on specific neurons.
Dense(10, activation='softmax'): The output layer has 10 neurons because we are classifying 10 digits (0-9).
The softmax activation ensures that the outputs are probabilities that sum to 1. The neuron with the highest probability will correspond to the predicted digit.

5. Compile the Model
Here, we specify how the model should be trained and evaluated:
optimizer='adam': Adam is an efficient optimization algorithm that adjusts learning rates during training. It's widely used for its speed and good performance.
loss='sparse_categorical_crossentropy': This is the loss function used to measure how well the model is doing. In classification tasks, categorical crossentropy is commonly used. Since our labels are integers (not one-hot encoded), we use the "sparse" version.
metrics=['accuracy']: Accuracy is used to evaluate the performance of the model during training and testing. It tells us the percentage of correctly predicted labels.

6. Train the Model
model.fit(): Trains the model on the training data (x_train, y_train).
epochs=5: The number of times the entire training dataset is passed through the model. In this case, we train for 5 epochs.
During each epoch, the model will update its weights based on the training data and minimize the loss function.

7. Evaluate the Model
model.evaluate(): After training, we evaluate the model on the test data (x_test, y_test).
test_loss: The loss on the test data.
test_acc: The accuracy on the test data.
verbose=2: Controls the level of detail in the output. Here, it gives a concise summary of the evaluation process.

8. Print Test Accuracy
After evaluating the model, we print the test accuracy to see how well it performed on unseen data. Accuracy is displayed as a percentage (rounded to 4 decimal places).
Summary of Model Flow:
Load data → 2. Preprocess data (normalize) → 3. Build model → 4. Compile model → 5. Train model → 6. Evaluate model
This code gives you a basic neural network that learns to classify handwritten digits using TensorFlow.
