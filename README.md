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

To easily compare the performance of my model with others, I have a comparison table below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/87942072/139209235-2d5d453f-8e43-4bb5-99ae-4e53cbee11c5.png" />
</p>

In this table is a comparison between my model and 2 models: VGG16-GRU and I3D, these two models I have obtained information from an article from australian national university and trained on dataset WLASL100 with 100 words in English.

Here we see that the top 1 highest accuracy of the other 2 models is ~66%, and it is lower than my accuracy which is 92%. I know it's really unfair to have a comparison between 11 classes and 100 classes but with the top 1 accuracy ~92%, this shows us that this model really has potential to grow in the future.

## Deploy with streamlit
In order to using streamlit to use this application, you need to install streamlit first:
- Install streamlit: `pip install streamlit`
- Run streamlit: `streamlit run streamlit.py`
- Result look like that:

<p align="center">
  <img src="https://user-images.githubusercontent.com/87942072/139210018-8463ec00-f20b-4045-912f-efa4f94132f2.png" />
</p>

## Future works:
In the future, if I have more time, I will:
- Training model in 1000 English common words
- Developing feature sign to sentence with NLP
- Deploying mobile phone application




