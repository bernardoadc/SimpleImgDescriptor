import requests
from PIL import Image
from transformers import BlipForConditionalGeneration, AutoProcessor
import gradio as gr

# Função para carregar o modelo
def load_model(model_name):
    if model_name == "Salesforce BLIP":
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    # elif model_name == "EVA-CLIP":
    #     processor = AutoProcessor.from_pretrained("ViTAE-Transformer/EVA")
    #     model = BlipForConditionalGeneration.from_pretrained("ViTAE-Transformer/EVA")
    # elif model_name == "EVCap":
    #     processor = AutoProcessor.from_pretrained("Jiaxuan-Li/EVCap")
    #     model = BlipForConditionalGeneration.from_pretrained("Jiaxuan-Li/EVCap")
    elif model_name == "BLIP-2 FlanT5-XXL":
        processor = AutoProcessor.from_pretrained("salesforce/blip2-flan-t5-xxl")
        model = BlipForConditionalGeneration.from_pretrained("salesforce/blip2-flan-t5-xxl")
    return processor, model

# Função para gerar a legenda
def generate_caption(image, model_name):
    processor, model = load_model(model_name)
    # Convert the image to RGB format
    image = image.convert('RGB')
    text = "the image of"
    inputs = processor(images=image, text=text, return_tensors="pt")

    # Generate a caption for the image
    outputs = model.generate(**inputs, max_length=50)

    # Decode the generated tokens to text
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

# Interface do Gradio
iface = gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Radio(["Salesforce BLIP", "BLIP-2 FlanT5-XXL"], label="Choose Model") # , "EVA-CLIP", "EVCap"
    ],
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Captioning with Multiple Models",
    description="Upload an image and choose a model to get a generated caption."
)

# Launch the interface
iface.launch()
