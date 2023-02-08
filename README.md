# JeopardyModel

The goal is to generate good pretrained models for multimodal use-cases. The [VQA dataset](https://visualqa.org/) gives us images, along with associated questions and answers (e.g. image of person surfing, Q: "What is the person doing?", A: "Surfing"). Most questions are of the type ("How many people are surfing?", A: "2"). 

But why are we calling this the Jeopardy model? This is because the wonderful game show Jeopardy! informs our pretraining strategy - we provide a concatenated answer-image embedding to the model, and expect a question in response.


To get up and running, make a conda virtual environment using the enviroment.yml file. 

`conda create -f environment.yml`

Install dependencies `pip3 install -r requirements.txt`

And get off to the races
