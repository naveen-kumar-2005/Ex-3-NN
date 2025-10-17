<!DOCTYPE html>
<html>
<body>

  <H3>ENTER YOUR NAME: Naveen Kumar.R Y</H3>
<H3>ENTER YOUR REGISTER NO.: 212223230139</H3>
<H3>EX. NO.3</H3>
  <h2>Implementation of MLP for a Non-Linearly Separable Data</h2>

  <h3>Aim:</h3>
  <p>To implement a Perceptron for classification using Python.</p>

  <h3>Theory:</h3>
  <p>Exclusive OR (XOR) is a logical operation that outputs true when the inputs differ. The XOR gate truth table is:</p>

  <img src="https://user-images.githubusercontent.com/112920679/195774720-35c2ed9d-d484-4485-b608-d809931a28f5.gif" alt="XOR Table">

  <p>XOR is a classification problem with binary outputs. If we plot the inputs against the outputs, we see:</p>

  <img src="https://user-images.githubusercontent.com/112920679/195774898-b0c5886b-3d58-4377-b52f-73148a3fe54d.gif" alt="XOR Plot">

  <p>This plot shows that the classes cannot be separated by a single straight line. We'd need multiple decision boundaries:</p>

  <img src="https://user-images.githubusercontent.com/112920679/195775012-74683270-561b-4a3a-ac62-cf5ddfcf49ca.gif" alt="Non-linearly Separable">

  <p>This limitation gave rise to the concept of **hidden layers**, which led to the development of **Multilayer Perceptrons (MLP)**:</p>

  <img src="https://user-images.githubusercontent.com/112920679/195775183-1f64fe3d-a60e-4998-b4f5-abce9534689d.gif" alt="MLP Architecture">

  <p>
    MLPs allow data to be processed in multiple dimensions through hidden layers. Each layer applies a weighted transformation and passes the output forward. This is known as **feedforward**. The network learns by updating weights via **backpropagation**, optimizing output predictions.
  </p>

  <h3>Algorithm:</h3>
  <ol>
    <li>Initialize input patterns for XOR Gate</li>
    <li>Initialize the desired output</li>
    <li>Initialize weights for the 2-layer MLP (2 hidden neurons, 1 output neuron)</li>
    <li>Repeat until the loss stabilizes:
      <ul>
        <li>Perform forward pass</li>
        <li>Calculate error</li>
        <li>Update weights using backpropagation</li>
        <li>Store loss values</li>
      </ul>
    </li>
    <li>Test the trained network with XOR patterns</li>
  </ol>

  <h3>Program:</h3>
  <pre>
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[0,0,1,1],[0,1,0,1]])
y = np.array([[0,1,1,0]])
n_x, n_y, n_h = 2, 1, 2
m = x.shape[1]
lr = 0.1

np.random.seed(2)
w1 = np.random.rand(n_h, n_x)
w2 = np.random.rand(n_y, n_h)
losses = []

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_prop(w1, w2, x):
    z1 = np.dot(w1, x)
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1)
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def back_prop(m, w1, w2, z1, a1, z2, a2, y):
    dz2 = a2 - y
    dw2 = np.dot(dz2, a1.T) / m
    dz1 = np.dot(w2.T, dz2) * a1 * (1 - a1)
    dw1 = np.dot(dz1, x.T) / m
    return dz2, dw2, dz1, dw1

iterations = 10000
for i in range(iterations):
    z1, a1, z2, a2 = forward_prop(w1, w2, x)
    loss = -(1/m) * np.sum(y * np.log(a2) + (1 - y) * np.log(1 - a2))
    losses.append(loss)
    dz2, dw2, dz1, dw1 = back_prop(m, w1, w2, z1, a1, z2, a2, y)
    w2 -= lr * dw2
    w1 -= lr * dw1

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.show()

def predict(w1, w2, input):
    _, _, _, a2 = forward_prop(w1, w2, input)
    result = 1 if np.squeeze(a2) >= 0.5 else 0
    print(f"Input: {[i[0] for i in input]}  Output: {result}")

print('--- Prediction Results ---')
predict(w1, w2, np.array([[1],[0]]))
predict(w1, w2, np.array([[1],[1]]))
predict(w1, w2, np.array([[0],[1]]))
predict(w1, w2, np.array([[0],[0]]))
  </pre>

  <h3>Output:</h3>
<img width="838" height="557" alt="image" src="https://github.com/user-attachments/assets/6ff4e431-97d2-4170-b8a2-65aa5f1c8689" />
<img width="524" height="129" alt="image" src="https://github.com/user-attachments/assets/968d4989-2d0b-4a58-ae54-753675398bf5" />

  <h3>Result:</h3>
  <p>The XOR classification problem was successfully solved using a Multilayer Perceptron (MLP) implemented in Python.</p>

</body>
</html>
