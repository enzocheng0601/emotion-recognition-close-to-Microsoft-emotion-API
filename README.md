# emotion-recognition-close-to-Microsoft-emotion-API
a VGG-16 based CNN model trained by the label gaining from Microsoft emotion API and implemented on TensorFlow

train a FER (Facial expression recognition) model close to Microsoft emotion API on tensorflow.

We use face_detection model, which is also implemented by tensorflow, to help us extract all faces in our dataset and then combine those faces, every 50 faces, into an image. Labeling those combined images through Microsoft emotion API and using these labels to train our model, this FER_model is based on VGG-16 CNN structure.

dependencies you will need for training your own FER model in your own dataset:
1. tensorflow: This model is built on tensorflow, make sure that you have already had tensorflow on your work platform before you continue the following steps. (GPU version will be preferred, saving lots of time)
2. install dependencies listed below through pip: cv2, pillow, matplotlib
3. You will need a platform which is allowed to run an .exe file (Windows should be the only one qualified): Since we need label our images through Microsoft emotion API, which we have already packed into an .exe file, thus, make sure that you can run it successfully.

how to train your own model in your own dataset:
1. git clone this repository to your computer

2. cd /to/the/directory/you/just/cloned/ 

3. run python training_and_val.py with 7 arguments:
        
        arguv[1]: path to your own dataset
        
        arguv[2]: path to the output of combined images
        
        arguv[3]: path to the output of label return by Microsoft emotion API (json format)
        
        arguv[4]: path to the emotion API .exe file ( /this/repository/emotion_api/FaceID.exe)
        
        arguv[5]: path to store training log
        
        arguv[6]: path to store validation log
        
        arguv[7]: number of iteration (should be an integer)
 
 How to evaluate the model:
  same as how you run the training part but with different arguments:
        
        arguv[1]: path to where you store training log
        
        arguv[2]: path to the test directory
        
        arguv[3]: output path for real test image which is preprocessed
        
        arguv[4]: output path for the label of test image
        
        arguv[5]: path to the emotion API .exe file ( /this/repository/emotion_api/FaceID.exe)
        
 Result:
 The following result is based on the model with 6341 training image and 1586 validation images with 600 training step, 32 images per batch: 
 
 ![alt text](https://github.com/enzocheng0601/emotion-recognition-close-to-Microsoft-emotion-API/blob/master/demo/result.png)
 
 The testing result, the accuracy, is pretty close to the result of training, which is about 59.8%.
 
 
 
