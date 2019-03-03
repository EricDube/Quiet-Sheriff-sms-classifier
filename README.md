# fastai-text-classifier-server
A simple Flask server for a fast.ai text classifier model. Included is the SMS classifier notebook which I used to train the model. I'll be adding details to the deploy steps.

Example:
```
0.0.0.0:8080/predict?sms=this%20is%20a%20text
```
```
{
  "accuracy": 0.9927310347557068, 
  "prediction": "good"
}
```

## Setup

### Train the model
1. Setup a jupyter notebook project with the `sms-classifier.ipynb` notebook.
2. Change the `hushed` directory name to something that suits your application.
3. Download the Kaggle sms spam dataset: https://www.kaggle.com/uciml/sms-spam-collection-dataset
4. (Optional) Add your own SMS collections and labels.
5. Train your text classifier, at the end of the notebook the model will be exported as `export.pkl`.
6. Copy the `export.pkl` file into the data directory of the flask app.

## Deploy wih AWS ECS

### Local build
You'll need to have docker installed on your machine to get through this part.
1. Build the docker image: `docker build -t TextClassifier:latest`
2. Run the docker image and make sure it works: `docker run -d -p 80:80 TextClassifier:latest`, it'll be running on `0.0.0.0`.

### Upload to ECR
You'll need an AWS account to get through this part.
1. Search for ECR on the AWS web console, and then create a repository which will hold your docker image.
2. Click on `view push commands` and follow the instructions to upload your image.
3. Follow Amazon's directions for deploying a docker container: https://aws.amazon.com/getting-started/tutorials/deploy-docker-containers/ . When you create the task definition paste in the image URI of the docker image you uploaded to ECR. 

## End Result
https://www.quietsheriff.com/
