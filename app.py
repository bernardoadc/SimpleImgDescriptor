import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# AutoProcessor
# This is a processor class that is used for preprocessing data for the BLIP model.
# It wraps a BLIP image processor and an OPT/T5 tokenizer into a single processor.
# This means it can handle both image and text data, preparing it for input into the BLIP model.

# BlipForConditionalGeneration
# This is a model class that is used for conditional text generation given an image and an optional text prompt.
# In other words, it can generate text based on an input image and an optional piece of text.
# This makes it useful for tasks like image captioning or visual question answering, where
# the model needs to generate text that describes an image or answer a question about an image.

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image, DONT FORGET TO WRITE YOUR IMAGE NAME
# This image can either be a local file or fetched from a URL.
img_path = "cidade.jpg"
# convert it into an RGB format
image = Image.open(img_path).convert('RGB')

# You do not need a question for image captioning
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")

# Generate a caption for the image
outputs = model.generate(**inputs, max_length=50)

# Decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)
# Print the caption
print(caption)
