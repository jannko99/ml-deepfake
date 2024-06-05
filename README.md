# ml-deepfake detection
---
## Project goal/Motivation
---
### Motivation
---
The proliferation of deepfake technology poses a significant threat to the authenticity of visual media. With advancements in machine learning, creating highly realistic fake images has become easier and more accessible. These deepfake images can be used maliciously to spread misinformation, manipulate public opinion, and even blackmail individuals. The motivation behind this project is to develop a robust machine learning model capable of distinguishing between deepfake images and real images, thus contributing to the fight against digital deception.

### Relevance
---
In today's digital age, the ability to trust visual information is paramount. Deepfake images undermine this trust and can have serious implications in various sectors, including media, politics, and personal security. By developing a reliable method to detect deepfake images, this project aims to mitigate the risks associated with these forgeries. This is particularly relevant as the tools to create deepfakes continue to evolve, making detection more challenging and crucial.

### Objectives
---
The primary objective of this project is to create an accurate and efficient machine learning model that can detect deepfake images. The model should be capable of analyzing images and identifying subtle inconsistencies that indicate manipulation. Ultimately, this project seeks to enhance the tools available for digital forensics and ensure the integrity of visual media.

## Data
---
- The dataset is not included in the repository because it is too large. You can download it from [Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images/data)
- Unzip the file in the [detection-model/data] directory
- Create the folder data if it doesn't already exist
- The folder structure should look like this:

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

## Model Training and Fine-Tuning
---
- You can find the trainings logs [here](detection-model/workspace/logs)

### Overview
---
This project involves the development and fine-tuning of three convolutional neural network (CNN) models to detect deepfake images. The models were trained on a dataset of real and deepfake images to learn and distinguish between the two classes. Below are the details of the models and the training processes.
Model Architectures

#### Model_v1:
---
Layers:
    Conv2D (16 filters, 3x3 kernel)
    MaxPooling2D
    Conv2D (32 filters, 3x3 kernel)
    MaxPooling2D
    Conv2D (16 filters, 3x3 kernel)
    MaxPooling2D
    Flatten
    Dense (256 units)
    Dense (1 unit, output)
Total Parameters: 7,393,252
Training Epochs: 20

#### Model_v2:
---
Layers:
    Conv2D (16 filters, 3x3 kernel)
    MaxPooling2D
    Conv2D (32 filters, 3x3 kernel)
    MaxPooling2D
    Dropout
    Conv2D (16 filters, 3x3 kernel)
    MaxPooling2D
    Dropout
    Flatten
    Dense (256 units)
    Dropout
    Dense (1 unit, output)
Total Parameters: 7,892,964
Training Epochs: 10

#### Model_v3:
---
Layers:
    Conv2D (16 filters, 3x3 kernel)
    MaxPooling2D
    Conv2D (32 filters, 3x3 kernel)
    MaxPooling2D
    Dropout
    Conv2D (16 filters, 3x3 kernel)
    MaxPooling2D
    Dropout
    Flatten
    Dense (256 units)
    Dropout
    Dense (1 unit, output)
Total Parameters: 7,892,964
Training Epochs: 20

### Training Process
---
**Model_v1:** Initially trained for 20 epochs. This model showed signs of overfitting, as indicated by the increasing validation loss and the significant gap between training and validation accuracy over the epochs.

- Epoch 1: Training Accuracy: 92.55%, Validation Accuracy: 90.08%, Validation Loss: 0.2375
- Epoch 20: Training Accuracy: 99.62%, Validation Accuracy: 92.22%, Validation Loss: 0.6815

**Model_v2:** To address overfitting, dropout layers were added. This model was trained for 10 epochs and demonstrated better generalization on the validation set compared to Model 1.

- Epoch 1: Training Accuracy: 71.90%, Validation Accuracy: 83.72%, Validation Loss: 0.3735
- Epoch 10: Training Accuracy: 93.26%, Validation Accuracy: 91.72%, Validation Loss: 0.1955

**Model_v3:** A third model was trained, incorporating dropouts and trained for 20 epochs to further optimize performance. This model balanced between complexity and regularization, providing improved accuracy and robustness.

- Epoch 1: Training Accuracy: 73.99%, Validation Accuracy: 84.71%, Validation Loss: 0.3391
- Epoch 20: Training Accuracy: 95.30%, Validation Accuracy: 92.75%, Validation Loss: 0.1759

### Training Logs
---
The training logs for each model, capturing the loss and accuracy per epoch, provide insights into the learning process and model performance over time. Here are the summarized logs for each model:

**Model_v1:**
Noticeable overfitting with increasing validation loss and minimal improvement in validation accuracy after initial epochs.

**Model_v2:**
Addition of dropout layers helped mitigate overfitting, resulting in better performance and lower validation loss.

**Model_v3:**
Further fine-tuning with dropout layers and extended training epochs resulted in the best performance among the three models, with a good balance between training and validation accuracy and reduced validation loss.

## Interpretation and validation of results 
---
After training three different models to detect deepfake images, the results showed varying degrees of success and challenges.

**Model_v1:**
- Training Performance: Model 1 achieved high training accuracy, peaking at 99.62% by epoch 20. However, the validation accuracy stagnated around 92.22%, and the validation loss increased significantly, indicating overfitting.

- Impact of Choices: The lack of regularization in Model 1 led to overfitting, where the model performed well on the training data but poorly generalized to unseen validation data. This highlighted the necessity for regularization techniques to improve generalization.

**Model_v2:**
- Training Performance: Introducing dropout layers in Model 2 improved the generalization ability. The validation accuracy reached 91.72% with a reduced validation loss of 0.1955 by epoch 10.

- Impact of Choices: The use of dropout layers effectively mitigated overfitting, as evidenced by the reduced gap between training and validation accuracy and lower validation loss. This underscored the importance of regularization in deep learning models.

**Model_v3:**
- Training Performance: Model 3, which included dropout layers and extended training to 20 epochs, showed the best overall performance. The validation accuracy was 92.75% with a validation loss of 0.1759.

- Impact of Choices: Extending the training duration allowed the model to better learn the dataset's intricacies, and the regularization helped maintain generalization, resulting in a well-balanced model with high accuracy and low loss.


### Test using the Testdata

**model-v1**

```
------------------
Precision: 87.76 %
Recall: 88.71 %
Accuracy: 88.28 %
```

**model-v2**

```
------------------
Precision: 87.21 %
Recall: 86.4 %
Accuracy: 86.99 %
```

**model-v3**

```
------------------
Precision: 87.35 %
Recall: 88.39 %
Accuracy: 87.92 %
```
The test results on the test dataset indicate that all three models perform similarly, with Model 1 showing slightly higher recall and accuracy, while Model 3 offers a balance between precision and recall.

### Diffusion Test
---

To further validate the models, a diffusion test was conducted by generating fake images using diffusion prompts and evaluating the models' ability to detect these fakes.

The images are generatet from [Diffison.to](https://diffusion.to/)

#### GPT-Promts for Fake images
---
**Man in a Formal Suit:**"A middle-aged man, about 35 years old, with short brown hair and a well-groomed beard. He is wearing a tailored dark blue suit with a light blue tie. His facial expression is friendly and confident, with blue eyes and a slight smile on his lips. The background is a blurred office environment."

**Woman with Natural Makeup:**"A young woman, about 25 years old, with long, straight blonde hair and light brown eyes. She is wearing light, natural makeup that highlights her beauty. Her facial expression is friendly and relaxed, with a radiant smile. She is wearing a plain white T-shirt, and the background is a blurred park on a sunny day."

**Man in Sportswear:**"A young man, about 28 years old, with short black hair and a muscular build. He is wearing a red sports T-shirt and black shorts. His facial expression is determined and focused, with brown eyes looking concentrated. The background shows a blurred stadium on a sunny day."

**Woman in Casual Clothing:**"A middle-aged woman, about 40 years old, with shoulder-length, wavy brown hair and green eyes. She is wearing a plaid shirt and blue jeans. Her facial expression is warm and friendly, with a hint of curiosity. The background is a cozy living room with soft, warm colors."

**Man with Distinct Features:**"An older man, about 50 years old, with short gray hair and a well-groomed full beard. His face is distinctive with deep wrinkles and sharp contours. He is wearing a dark gray T-shirt and a leather jacket. His facial expression is serious and contemplative, with deep gray eyes. The background is a blurred urban setting at dusk."

#### Diffusion Test Results
---
Detect the Fake images.

Fake: Prediction > 0.5
Real: Prediction < 0.5

**Test result model_v1**
```
--------------------
Prediction: 1.0
Prediction: 0.008749807253479958 #Wrong
Prediction: 0.9999996423721313
Prediction: 1.0503980547582614e-06 #Wrong
Prediction: 0.9996607899665833
```


**Test result model_v2**
```
--------------------
Prediction: 0.8086673021316528
Prediction: 0.4395149052143097 #Wrong
Prediction: 0.7271615266799927
Prediction: 0.021419500932097435 #Wrong
Prediction: 0.739109456539154
```


**Test result model_v3**
```
--------------------
Prediction: 0.9951998591423035
Prediction: 0.10734300315380096 #Wrong
Prediction: 0.9725474715232849
Prediction: 0.7071648240089417
Prediction: 0.9773142337799072
```
### Validation of Results
---
The models were evaluated against the diffusion-generated fake images to test their robustness in detecting more challenging fakes. Model 3 showed the most consistent performance, correctly identifying most of the fake images with high confidence scores.

### Discussion

The training and validation results demonstrate that the specific choices made during model development had a significant impact on performance. Regularization techniques such as dropout were essential in preventing overfitting and improving generalization. Additionally, extending the training epochs allowed for better model optimization, balancing accuracy and loss.

The validation process, including the test dataset and diffusion test, highlighted that Model 3 is the most robust and consistent in detecting deepfake images. It performs competitively with a precision of 87.35%, recall of 88.39%, and accuracy of 87.92% on the test dataset and showed strong results in the diffusion test as well.

These findings underscore the importance of model fine-tuning and regularization in developing effective machine learning models for deepfake detection. The project successfully achieved its goal of creating a robust model for detecting deepfake images, contributing to the broader effort to combat digital misinformation.

## Deployment
---
- We use TensorFlow to create our model.
- If you want to rebuild the model, please refer to the TensorFlow [installation](https://www.tensorflow.org/install/pip#linux) guide.
- You can train the model with either CPU or GPU, although GPU training is faster.

### detection-model
---
- The model is programmed in a Jupyter notebook.
- You can set it up in Google Colab.
- Alternatively, for a local setup, I recommend using a Conda environment.
- You can find the requirements [here](detection-model/requirements/).
- The most important installation is **TensorFlow**.

### webapp-backend
---

- For using the model, there is a WebApp that you can deploy in a Docker container.
- To replace your model, place the `*.keras` file in this  [folder](webapp-backend/workspace/model/). Don't forget to adjust the model path in the [app.py](webapp-backend/workspace/app.py) file.
- To exchange your logs for viewing in **TensorBoard**, place the log files [here](webapp-backend/workspace/logs/tensorboard_logs/).

**Build the container:**

- Navigate to the directory where the [Dockerfile](webapp-backend/dockerfile) is located.

```bash
docker build -t flask-deepfake-app .
```
```bash
docker run -p 80:80 -p 6006:6006 flask-deepfake-app
```
- Then, you can access the web app on localhost under port 80.
- You can also view the training logs on TensorBoard.
- When accessing TensorBoard logs, navigate back to the navbar, select Model, and then return to logs to see TensorBoard. This is necessary because the TensorBoard server needs time to start up and it may throw an error on the first visit to the TensorBoard logs page.

### webapp-frontend
---
- It's a Vue.js frontend.
- You can improve it, but you don't have to make any changes in this folder.
