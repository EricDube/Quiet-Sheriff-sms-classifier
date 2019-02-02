# fastai-text-classifier-server
A simple Flask server for a Fast.ai text classifier model. Included is the SMS classifier notebook which I used to train my moodel.

Example:
```
0.0.0.0:8080/predict?sms=this%20is%20a%20text
```
```
{
  "accuracy": "tensor([3.9485e-02, 1.0045e-04, 9.6041e-01])", 
  "prediction": "good"
}
```

## Setup

1. Train your text classifier and export the model.
```
learner.export()
```
2. Copy the `export.pkl` file into the data directory.

## Deploy
1. Follow the Fast.ai instructions for AWS Beanstalk: https://course.fast.ai/deployment_aws_beanstalk.html
