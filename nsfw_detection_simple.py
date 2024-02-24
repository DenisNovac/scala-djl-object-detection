# https://huggingface.co/Falconsai/nsfw_image_detection

import torch
from PIL import Image
from transformers import pipeline

images = (
  "images/pony-toys.jpg",
  "images/street.jpg",
  "images/nudity.jpg",
  "images/hentai.jpg",
  "images/dog_bike_car.jpg"
)

model_name = "Falconsai/nsfw_image_detection"
#model_name = "converted.pt"

classifier = pipeline("image-classification", model=model_name)

def check(image_name):
    img = Image.open(image_name)
    r = classifier(img)
    print("{0} -- {1}".format(image_name, r))


result = map(lambda n: check(n), images)

# maps are lazy in python, need to call list() to run
list(result)

# images/pony-toys.jpg -- [{'label': 'normal', 'score': 0.9974840879440308}, {'label': 'nsfw', 'score': 0.0025159099604934454}]
# images/street.jpg -- [{'label': 'normal', 'score': 0.9998045563697815}, {'label': 'nsfw', 'score': 0.00019550778961274773}]
# images/nudity.jpg -- [{'label': 'nsfw', 'score': 0.9998843669891357}, {'label': 'normal', 'score': 0.00011558941332623363}]
# images/hentai.jpg -- [{'label': 'nsfw', 'score': 0.9997820258140564}, {'label': 'normal', 'score': 0.00021801472757942975}]
# images/dog_bike_car.jpg -- [{'label': 'normal', 'score': 0.9998995065689087}, {'label': 'nsfw', 'score': 0.00010048435069620609}]

