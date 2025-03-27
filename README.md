# Developing CFD Surrogate Models Using Machine Learning for A-Coil Velocity Profile Prediction
This project uses ML models (e.g. ANN, KNN, GaussianProcess, RandomForest, DecisionTree and LinearRegression) to develop computionally efficient suggorate model with CFD simulation data.
The suggorate model can be used for the design optimization of A-coil or HVAC system.

## Generate CFD data

Firstly, we solved CFD model parametrically and output data for data analysis by using ANSYSFLEUNT-Python API: Pyfluent. The data processing flow is shown in below:

(File: PyFluent.py)

![PyFluent](https://github.com/PochingHsu/Acoil/assets/165426535/f490f5a3-243e-489c-8ffa-9543a6080d48)

## ML suggorate Models Development
The air velocity profile prediction of ANN model:

(File: DataSplit.py -> ann.py -> ann_pred.py)
 
![ANN](https://github.com/PochingHsu/Acoil/assets/165426535/7ee27fa2-cc4b-4f17-999d-3811acadbe17)

The air velocity profile prediction of GaussianProcess model:

(File: DataSplit.py -> ml.py -> ml_pred.py)
 
![GP](https://github.com/PochingHsu/Acoil/assets/165426535/8b564037-fcb9-4443-a989-cbf6c3d2a4e4)
