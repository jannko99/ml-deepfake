# ml-deepfake detection

## Project goal/Motivation

## Training

You can find the Trainings logs here [detction-model/workspace/logs]

## Interpretation and validation of results 



### model-v1

### model-v2

### model-v3

## Data

- The Dataset is not in the repository, becaus its to large. You can download it from [Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images/data)
- Unzip the File here [detection-model/data]
- Create the folder **data** if it's not there
- Folderstructur schould look like that

```
workspace/data/
├── Dataset
│   ├── Test
│   │   ├── Fake
│   │   └── Real
│   ├── Train
│   │   ├── Fake
│   │   └── Real
│   └── Validation
│       ├── Fake
│       └── Real
└── placeholder.txt
```

## Deployment
---

- We use Tensorflow to creat our model
- If you want to rebuild the model, you should have a look at the tensorflow [installation](https://www.tensorflow.org/install/pip#linux)
- You can train the model with CPU or GPU, doesnt matter. GPU is faster.

### detection-model
---
- The model is programed on a jupyternotebooke
- So you can set it up in google colab

### webapp-backend
---

docker build -t flask-deepfake-app .

docker run -p 80:80 -p 6006:6006 flask-deepfake-app
