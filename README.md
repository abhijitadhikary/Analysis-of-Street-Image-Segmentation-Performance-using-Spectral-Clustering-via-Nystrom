### Project Structure ###

To run in the default settings please follow the following steps:

1. Paster the `idd20k_lite.zip` file in the `./data` ditectory.
2. Go into the `./src/` directory and run `python main.py` to run the project using default hyperparameters.
3. Hyperparameters can be changed from the `./src/utils.py` file's `get_args()` function.

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