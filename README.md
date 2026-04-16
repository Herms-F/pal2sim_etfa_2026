# IMU-Based Object Activity Recognition:
## An Industrial Dataset and Benchmark Study for Enhanced Process Transparency in Logistics
### Review code

This repository contains the code for running trainings on SPARL3 dataset, an industrial dataset for object activity recognition using IMU sensors. The dataset and code are designed to facilitate research in process transparency in logistics.

As the underlying classificator LSTM are used for the publication, please refer to the original paper for more details.
The repo also contains different models (2-stage LSTM, and CNN) for testing.

The read_data.py file contains the initial data loading and preprocessing functions.

To just run the training, execute the main.py file in the lstm folder. 
You can modify the parameters in the config.py file to customize the training process.

To build the environment, you can use the provided requirements.txt file (tested with python3.12)