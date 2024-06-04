# ml-deepfake detection

## Project goal/Motivation

- Goal is to Detect deepfake images from real images

## Training

- You can find the Trainings logs [here](detction-model/workspace/logs)

### model-v1
---

### model-v2
---

### model-v3
---

## Interpretation and validation of results 
---


### model-v1
---

### model-v2
---

### model-v3
---

## Data
---
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
- The model is programed on a jupyternotebook
- So you can set it up in google colab
- Or localy i recomend a conda enviroment
- You can find the requiremnts [here](detection-model/requirements/)
- Most important install is **Tensorflow**

### webapp-backend
---
- For usage of the moder there is a WebApp, you can deploy the app in a docker container
- To Exchang your model, you can put the `*.keras` file in this [folder](webapp-backend/workspace/model/). Don't forget to adjust the modelpath in the [app.py](webapp-backend/workspace/app.py) file
- To Exchang your logs, to lookup them in the **tensorboard**, you can put the logfiles [here](webapp-backend/workspace/logs/tensorboard_logs/)

Build the container:

- Got to the directory where the [dockerfile](webapp-backend/dockerfile) is locatet

```bash
docker build -t flask-deepfake-app .
```
```bash
docker run -p 80:80 -p 6006:6006 flask-deepfake-app
```
- Then you can visit on localhost under port 80 the webapp
- You also can lookup the training logs on tensorboard
- When you go to **Tensoboard logs**, you have to go on time back to **Model** and the you can see the Tensoboard. This is becaus the Tensoboard server need to startup.

### webapp-frontend
---
- It's a vue.js frontend
- You can improve it, but you don't have to do anything in this folder
