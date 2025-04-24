# Usage 

To launch the application use the information provided [here](./readme.md)

## Installation

TBW

## Candidates and templates

This should be the architecture of the `static` folder

```bash
static
|--- <DATASET>
|--- |--- templates
|--- |--- |---- 0.png
|--- |--- |---- 1.png
...
|--- |--- candidates
|--- |--- |---- <MODELNAME-1>_0.png
|--- |--- |---- <MODELNAME-1>_1.png
|--- |--- |---- <MODELNAME-2>_0.png
|--- |--- |---- <MODELNAME-2>_1.png
...
```

Where the `<NAME>` notation can be defined by the user.

In the `app.py` file, make sure to properly define the dataset to use `DATASET="name_of_the_folder"`, and the `MODEL_IDS=["model-name-1", "model-name-2"]`
