# DEEP RULE VISUALIZER
### A Novel Explainable Deep Neural Network Visualization Architecture
This research project was undertaken to fulfill the requirement for the MSc in Artificial Intelligence at the University of Moratuwa.

Supervisor: Dr. Subha Fernando

![alt text](https://raw.githubusercontent.com/lahirukavinda/deep-rule-visualizer/main/utils/readme-image.png)

Deep Rule Visualizer is a novel architecture which aims to enhance the transparency and interpretability of deep neural networks. Deep Rule Visualizer incorporates cutting-edge techniques in explainable AI and visualization to provide users with insightful explanations for the decisions made by deep learning models. The framework leverages the xDNN architecture and advanced visualization tools to generate IF ... THEN rules and saliency maps, enabling users to understand the reasoning behind predictions. Additionally, a novel metric called Deep Rule Score is introduced to quantify the rule quality and facilitate the evaluation of the framework's performance. Experimental results on benchmark datasets demonstrate the effectiveness of the Deep Rule Visualizer in improving model interpretability, fostering trust, and enabling better decision-making in critical domains.

### Steps
1. Update `.env` file with relevant local data set locations <br/>
   (For MNIST, images will be downloaded in the specified directory) <br/>
3. Execute `./run.py -d mnist` <br/>
or in PyCharm `Run > Edit Configurations > Parameters` add `-d mnist`

### References
xDNN: [https://github.com/Plamen-Eduardo/xDNN---Python](https://github.com/Plamen-Eduardo/xDNN---Python) <br />
Deconvnet/ Occlusion : [https://github.com/saketd403/Visualizing-and-Understanding-Convolutional-neural-networks](https://github.com/saketd403/Visualizing-and-Understanding-Convolutional-neural-networks) <br />
MNIST Data Download : [Yann LeCun's MNIST data through TensorFlow](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.7/tensorflow/g3doc/tutorials/mnist/download/index.md) <br />
