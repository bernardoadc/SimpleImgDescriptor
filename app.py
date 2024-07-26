import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import gradio as gr

# Carregar o processador e o modelo pré-treinados
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    # Convert the image to RGB format
    image = image.convert('RGB')
    text = "the image of"
    inputs = processor(images=image, text=text, return_tensors="pt")

    # Generate a caption for the image
    outputs = model.generate(**inputs, max_length=50)

    # Decode the generated tokens to text
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

# Create a Gradio interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Captioning with BLIP",
    description="Upload an image and get a generated caption using the BLIP model."
)

# Launch the interface
iface.launch()
