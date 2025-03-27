# Developing CFD Surrogate Models Using Machine Learning for A-Coil Velocity Profile Prediction
This project leverages machine learning models (e.g., ANN, KNN, Gaussian Process, Random Forest, Decision Tree, and Linear Regression) to develop a computationally efficient surrogate model trained on CFD simulation data. The surrogate model can be used for design optimization of A-coils and HVAC systems.

##  Automated CFD Data Generation

First, I use ANSYS Fluent-Python API, PyFluent to automatically and parametrically solved the CFD models, and then extracted output data for data analysis and training ML surrogate models. The data processing workflow is shown below:

(File: PyFluent.py)

![PyFluent](https://github.com/PochingHsu/Acoil/assets/165426535/f490f5a3-243e-489c-8ffa-9543a6080d48)

## ML suggorate Models Development
The air velocity profile prediction of ANN model:

(File: DataSplit.py -> ann.py -> ann_pred.py)
 
![ANN](https://github.com/PochingHsu/Acoil/assets/165426535/7ee27fa2-cc4b-4f17-999d-3811acadbe17)

The air velocity profile prediction of GaussianProcess model:

(File: DataSplit.py -> ml.py -> ml_pred.py)
 
![GP](https://github.com/PochingHsu/Acoil/assets/165426535/8b564037-fcb9-4443-a989-cbf6c3d2a4e4)
