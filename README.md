# Airbus ship image segmentation

## Task
You are required to locate ships in images, and put an aligned bounding box segment around the ships you locate. Many images do not contain ships, and those that do may contain multiple ships. Ships within and across images may differ in size (sometimes significantly) and be located in open sea, at docks, marinas, etc.

For this metric, object segments cannot overlap. There were a small percentage of images in both the Train and Test set that had slight overlap of object segments when ships were directly next to each other. Any segments overlaps were removed by setting them to background (i.e., non-ship) encoding. Therefore, some images have a ground truth may be an aligned bounding box with some pixels removed from an edge of the segment. These small adjustments will have a minimal impact on scoring, since the scoring evaluates over increasing overlap thresholds.

## Stack Technologies

-   Python
-   Keras
-   Numpy, Pandas, Matplotlib
-   Streamlit

## Installation

### Clone project
```bash
git clone https://github.com/DHushchin/airbus-ship-detection
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Load [dataset](https://www.kaggle.com/competitions/airbus-ship-detection/data) to /data directory
```
kaggle competitions download -c airbus-ship-detection
```
## Deploy
https://dhushchin-airbus-ship-detection-srcapp-d6ihp9.streamlitapp.com/

