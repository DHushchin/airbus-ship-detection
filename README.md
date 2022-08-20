# Airbus ship image segmentation

## Task
You are required to locate ships in images, and put an aligned bounding box segment around the ships you locate. Many images do not contain ships, and those that do may contain multiple ships. Ships within and across images may differ in size (sometimes significantly) and be located in open sea, at docks, marinas, etc.

For this metric, object segments cannot overlap. There were a small percentage of images in both the Train and Test set that had slight overlap of object segments when ships were directly next to each other. Any segments overlaps were removed by setting them to background (i.e., non-ship) encoding. Therefore, some images have a ground truth may be an aligned bounding box with some pixels removed from an edge of the segment. These small adjustments will have a minimal impact on scoring, since the scoring evaluates over increasing overlap thresholds.

## Stack Technologies

-   Python
-   Keras
-   Numpy, Pandas, Matplotlib
-   Streamlit

## Project structure
Source code contains code for [exploratory data analysis](./src/EDA/EDA.ipynb), 
model [training](./src/training) and [interference](./src/inference).
[Model directoty](./model) contains pre-trained model and its weights.
Also, there is [app.py](./src/app.py) script is an entry point of the streamlit application.
streamlit folder contains [configuration](./.streamlit/config.toml) for the application.

## Project tree

 * [.streamlit](./.streamlit)
     * [config.toml](./.streamlit/config.toml)
 * [model](./model)
     * [model.h5](./model/model.h5)
     * [weights.h5](./model/weights.h5)
 * [src](./src)
     * [EDA](./src/EDA)
         * [EDA.ipynb](./src/EDA/EDA.ipynb)   
     * [inference](./src/inference)
         * [inference.py](./src/inference/inference.py)
         * [utils.py](./src/inference/utils.py)
     * [training](./src/train)
         * [model.py](./src/training/model.py)
         * [train.py](./src/training/train.py)
         * [utils.py](./src/training/utils.py)
     * [app.py](./src/app.py)
     * [constants.py](./src/constants.py)
     * [img_processing.py](./src/img_processing.py)
 * [LICENSE](./LICENSE)
 * [README.md](./README.md)
 * [requirements.txt](./requirements.txt)
 * [.gitignore](./.gitignore)
 * [.gitattributes](./.gitattributes)
 

## Installation

### Clone project
```bash
git clone https://github.com/DHushchin/airbus-ship-detection
```

### Create virtual environment
```bash
python -m venv venv
```

### Activate it (depends on the OS)
```bash
venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Load [dataset](https://www.kaggle.com/competitions/airbus-ship-detection/data) to /data directory
```bash
kaggle competitions download -c airbus-ship-detection
```

### Run streamlit app locally 
```bash
streamlit run src/app.py
```
