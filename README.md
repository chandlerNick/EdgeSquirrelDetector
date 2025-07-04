# EdgeSquirrelDetector

In this repo, I examine edge computer vision methods to detect two species of squirrel that often disrupt birdfeeders in Washington. These squirrels are the Eastern Gray Squirrel (Sciurus carolinensis) and the Western Gray Squirrel (Sciurus griseus) most commonly.

## Data

All of size 224 x 224

Raw Data:
- 500 images - Sciurus carolinensis - Positive
- 500 images - Sciurus griseus - Positive

- 500 images - Robins - Negative
- 500 images - Chickadees - Negative
- 500 images - Sparrows - Negative
- 500 images - Crows - Negative

- 1000 positive
- 2000 negative

Data Augmentation, for each image:
- rotate 180 degrees
- rotate 90 degrees
- rotate 270 degrees

Total counts:
- 3000 positive
- 6000 negative

To get the dataset, run:

`uv run python3 src/build_dataset.py`

`uv run python3 src/resize_dataset.py`

`uv run python3 src/augment_dataset.py`

## Models



## Results

In the `models` directory, you can access both the full and edge-ready models.

## Plan:

### 1. Figure out hardware to use

Raspberry pi 3 b+ with v2 camera (train a model to work on the 5 too)

Note specs of each: processors, ram, camera resolution

### 2. Figure out software to use

Check for models that are complex enough to work but small enough to run

### 3. Figure out data to use

Find the type(s) of squirrel that occur(s) often in Washington

### 4. Get/annotate data

Check iNaturalist for some labeled data, label it with label studio?

### 5. Implement (train) software

Get cluster workflow setup, train on cluster

### 6. Test on the edge hardware

Make a simple test script that lights an LED when a squirrel is detected.

### 7. Enjoy your results

Yay.
