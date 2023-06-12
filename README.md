Simple App for image generation :smiley:
====

In this repo, we used Jax implementation of VAE together with deployment using fastapi, streamlit and docker to make a simple app for generating number images. The pipeline is quite clear so that everyone can utilize this to make better application :smile:  

In order to run app, just make sure that you have Docker in your computer and clone this repo. 
Then:
```
cd Image-Generation-App
docker-compose up -d --build
``` 

The app is already on `localhost:8501`. I already run the app on EC2 AWS and public here: `imageapp.maxph-realmlops.com`. Maybe I will not set it for a long time, so enjoy 🥲
