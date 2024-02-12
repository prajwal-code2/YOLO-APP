#  <p align ="center" height="40px" width="40px"> 💻YOLO APP💻 </p>

#  <p align ="center" height="40px" width="40px">  🤖App that utilizes the power of YOLOv8🤖 </p>



### <p align ="center"> Implemented using: </p>
<p align ="center">
<a href="https://www.python.org/" target="_blank" rel="noreferrer">   <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/800px-Python-logo-notext.svg.png" width="64" height="64" /></a>
<a href="https://opencv.org/" target="_blank" rel="noreferrer">   <img src="https://opencv.org/wp-content/uploads/2022/05/logo.png" width="64" height="64" /></a> 
<a href="https://www.tensorflow.org/" target="_blank" rel="noreferrer">   <img src="https://github.com/prajwal-code2/dogs-emotion/assets/74657725/92956e76-9975-48fe-b253-66073a64aa58" width="128" height="80" /></a>
<a href="https://keras.io/" target="_blank" rel="noreferrer">   <img src="https://github.com/prajwal-code2/dogs-emotion/assets/74657725/ed5ce50e-409e-4d3a-b4fc-f0853efb1bf9" width="128" height="64" /></a>  
</p>

<br>

##     <p align = "left"> Introduction 📚 </p>

Many dog owners have to leave their pets alone at home and have to regularly check how their dog is which is not an easy task. This project is a real-time dog emotion detector which tells whether dog is happy, sad, relaxed or angry. With this project, dog owner can be notified when their dog is not feeling well or is angry because someone breaks into the house and then he/she can take a suitable step. The model used in the project is built from scratch on dog-emotion dataset available on kaggle.  

<br>

##     <p align = "left">About CNN_Model💻 </p>

The system utilizes the model made from scratch. <br>Model uses 4 convolutional block(each block contains 2 Conv2D layers followed by a maxpooling layer) where filters used in blocks are 64, 128, 256 and 512 resp. after which a flatten layer is used which make the output of last convolutional block one-dimentional. Then 3 dense layers of 256, 128, 64 units resp. is used followed by a dense layer of 4 units and activation softmax to determine the emotion from the frame.<br> Diagram for illustration is given below:<br>

![dog_emotion](https://github.com/prajwal-code2/dogs-emotion/assets/74657725/8fda3d39-9e79-4417-850e-d131af2b52b2)

<br>

![model_architecture](https://github.com/prajwal-code2/dogs-emotion/assets/74657725/5ea92473-d63b-46d4-840e-f1a930a337cc)


<br>

##     <p align = "left"> Features ⭐ </p>
 -  Real-time emotion detector for dogs.
 -  Can be used for other animals as well after few modification.

<br>

##     <p align = "left"> Installation and Usage 🛠️ </p>
1. Clone this repository to your local machine .
2. Download the kaggle dog-emotion dataset from the link given in 'Dog Emotion' folder in the repository.
3. Run each cell of emotion_detector.ipynb and save the model in file after training is completed. 
4. Place the camera so that your dog is visible.
5. Run the script 'python use_model.py'.
6. The tool will then use your computer's camera to monitor the dog emotion.

<br>

##     <p align = "left"> DEMO Video 📹 </p>

<br>




https://github.com/prajwal-code2/YOLO-APP/assets/74657725/eb320695-006e-45ca-b9b9-0306086283ec





<br>

### <p align ="center"> Do remember to star ⭐ the repository if you like what you see!</p>

---


<div align="center">
  Made by <a href="https://github.com/prajwal-code2">Prajwal Joshi</a>
</div>
