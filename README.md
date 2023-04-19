# neural_networks
This repo contains all my work while learning neural networks 

<h1>Files:</h1>
<details><summary><h2>basic_nn.py</h2></summary>
<p> This is a basic neural network with 3 hidden layers. It uses backpropagation to update the weights and sigmoid activation as a threshold function</p>
<p>The weights are initialized randomly using the relation: </p>
<p align="center">-1/sqrt(number_of_nodes) to 1/sqrt(number_of_nodes)</p>
<p> During training, the calculated error (target-desired) is backpropagated using the relation:</p>
<p align="center">&#x3B4;E/&#x3B4;W<sub>jk</sub> = -(t<sub>k</sub> - O<sub>k</sub>) &#x2022; sigmoid(&#8721;<sub>j</sub>W<sub>jk</sub> &#x2022; O<sub>j</sub>)(1-sigmoid(&#8721;<sub>j</sub>W<sub>jk</sub> &#x2022; O<sub>j</sub>) &#x2022; O<sub>j</sub></p>
<h4>Block Diagram of Neural Network</h4>
<img>![image](https://user-images.githubusercontent.com/37641675/233109362-93c8738c-55b0-4497-8127-202af5885720.png)</img>

<h4>Requirements</h4>
<li>numpy</li>
<li>scipy</li>

<h4>Usage</h4> 
<li>Clone the repo</li>
<li>Run the requirements.txt file using the command: pip install -r requirements.txt</li>
<li>Run the basic_nn.py file with a path to the folder containing the number to be identified</li>
<li>This will train with the digits dataset provided and identify the numbers using the default parameters. To customize the parameters, run the basic_nn.py with the arguments: num_hidden_layers=xxx num_hidden_nodes=xxx learning_rate=xxx epochs=xxx</li>
<li> This will run and show the predicted values of the input</li>
</details>
