# https://huggingface.co/Falconsai/nsfw_image_detection

import torch
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor

images = (
  "images/pony-toys.jpg",
  "images/street.jpg",
  "images/nudity.jpg",
  "images/hentai.jpg",
  "images/dog_bike_car.jpg"
)

model_name = "Falconsai/nsfw_image_detection"
#model_name = "converted.pt"

model = AutoModelForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

def check(image_name):
    img = Image.open(image_name)

    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_label = logits.argmax(-1).item()
    model.config.id2label[predicted_label]

    is_nudity = predicted_label == 1
    print("{0} -- nsfw: {1}".format(image_name, is_nudity))


result = map(lambda n: check(n), images)

# maps are lazy in python, need to call list() to run
list(result)

# images/pony-toys.jpg -- nsfw: False
# images/street.jpg -- nsfw: False
# images/nudity.jpg -- nsfw: True
# images/hentai.jpg -- nsfw: True
# images/dog_bike_car.jpg -- nsfw: False
