# ASL_SignWish 
**Updating Readme is in progress**

<p align="center">
  <img src="https://user-images.githubusercontent.com/87942072/138586745-50618668-598b-425b-9348-8d2819c86692.png" />
</p>


## Motivation
According to the World Health Organization (WHO), there are about 450 million people who are deaf and have difficulty hearing. The only way they can communicate with each other is through sign language.

However, sign language is not really popular in the community of ordinary people, this makes it difficult for the deaf-mute community to access public services and communicate with ordinary people as well as career development.

Therefore, SignWish was born to make it easier for the deaf community to communicate and lead a better life.

## Dataset and Feature Extraction

Dataset to use in this project is manually trained by myself. For the first version, I collect 11 classes that are common words that we usually use with processing like the image below

For future versions, I will try to make 1000 common words in English

<p align="center">
  <img src="https://user-images.githubusercontent.com/87942072/138586643-b8d67cb1-fe5c-42d2-a5ab-5d14d7af8578.png" />
</p>

I collect data from my laptop webcam. And every frame I collected, I get landmarks of 2 hands and poses by using Mediapipe library. As a result we have total of 258 values from landmarks of hands and pose (coordinates).

## Model

After collecting data, the next step is trainning moel. I did experiment some type of models, and finally I personal build a Deep Learning model with LSTM layers and Dense layers

<p align="center">
  <img src="https://user-images.githubusercontent.com/87942072/139185931-94500e97-c92a-48b8-8ecb-ecc6afed7d51.png" />
</p>

With this manual build model, it achieved 92.05% of Top 1 accuracy. Not bad result!
