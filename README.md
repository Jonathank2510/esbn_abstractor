# Data Efficient Language Modeling using Emergent Symbol Binding

In this project we reimplement the Emergent Symbol Binding Network and then try to use it to augment a transformer with it. We want to see if introducing a seperate purely symbolic processing stream into a language model can increase performance on the scan dataset
The different architectures used in this project can be found in the Models folder. The reimplementation of the experiment by (Webb 2023) is done in the reimplementation.ipynb notebook. The experiments on the scan task can be found in the scan_experiment.ipynb notebook. Data from the training runs can be analysed via tensorboard and it is stored in the logs folder.
