### Running instructions ###

To run in the default settings please follow the following steps:

1. Paster the `idd20k_lite.zip` file in the `./data` ditectory.
2. Go into the `./src/` directory and run `python main.py` to run the project using default hyperparameters.
3. Hyperparameters can be changed from the `./src/utils.py` file's `get_args()` function.

*** It is recommended to run the project in a Windows environment as 
all the functionalities were tested in a Windows machine.
During the final testing phase, we found that the in a mac environment 
the system reads in images and ground truth labels in a different order 
than windows. This results in a mismatch between corresponding images, 
and the ground truth labels. *** 

### Directory Structure ###
```
.
├── data
│   └── *                   <- put the idd20k_lite.zip here before running
├── output
    └── stacked             <- side by side images (Image, Labels (GT), Labels (Pred))
        |───── train
        |───── val
        └───── test
├── documents
│   ├── docs
│   └── references
├── notebooks               <- notebooks for explorations / prototyping
└── src                     <- all source code, internal org as needed
```