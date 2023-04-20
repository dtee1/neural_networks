# neural_networks
This repo contains all my work while learning neural networks 

<h1>Files:</h1>
<details><summary><h2>basic_nn.py</h2></summary>
<p> This is a basic neural network with 1 hidden layer. It uses backpropagation to update the weights and sigmoid activation as a threshold function</p>
<p>The weights are initialized randomly using the relation: </p>
<p align="center">-1/sqrt(number_of_nodes) to 1/sqrt(number_of_nodes)</p>
<p> During training, the calculated error (target-desired) is backpropagated using the relation:</p>
<p align="center">&#x3B4;E/&#x3B4;W<sub>jk</sub> = -(t<sub>k</sub> - O<sub>k</sub>) &#x2022; sigmoid(&#8721;<sub>j</sub>W<sub>jk</sub> &#x2022; O<sub>j</sub>)(1-sigmoid(&#8721;<sub>j</sub>W<sub>jk</sub> &#x2022; O<sub>j</sub>) &#x2022; O<sub>j</sub></p>
<h4>Block Diagram of Neural Network</h4>
  
![image](https://user-images.githubusercontent.com/37641675/233166878-8ef3d47f-2c27-4944-b58d-9a0f9321976e.png)

<h4>Requirements</h4>
<li>numpy</li>
<li>scipy</li>

<h4>Usage</h4> 
<li>Clone the repo</li>
<li>Run the requirements.txt file using the command: pip install -r requirements.txt</li>
<li><strong>You can download the enitre MNIST Digits train dataset is csv format</strong></li>
<li>Run the basic_nn.py file with a path to the folder containing the number to be identified</li>
<li>This will train with the digits dataset provided and identify the numbers using the default parameters. To customize the parameters, run the basic_nn.py with the arguments: --hidden-nodes=xxx --learning-rate=xxx --epochs=xxx</li>
<li>This will run and show the predicted values of the input</li>

<h4>Suggested Improvements</h4> 
<li>Add more hidden layers: This has to be done with caution however as it could lead to overfitting and vanishing gradients</li>
<li>Find optimal number of epochs, learning rate, and nodes through simulation</li> 
<li>This neural network was done for single digit classification, however it can be extended to other tasks provided the dataset for training is available. For example, for a binary classification, we know the output nodes will be 2.</li>

<h4>Conclusion</h4>
<p>I am open to suggestions and improvements</p>
 </details>
