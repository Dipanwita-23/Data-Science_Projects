import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import weaviate
import base64
from transformers import CLIPProcessor, CLIPModel
import torch
from urllib.parse import urljoin
import os
import google.generativeai as genai
import gradio as gr
from langchain_core.messages import HumanMessage

from langchain_google_genai import ChatGoogleGenerativeAI
import weaviate.classes.config as wc
import json

client = weaviate.Client("http://localhost:8080")
chatbot = []
# Configuration
os.environ['GOOGLE_API_KEY'] = "API_KEY"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Functions for Data Scraping and Processing
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return image

def truncated_text(text, max_length=10):
    words = text.split()
    if len(words) > max_length:
        text = " ".join(words[:max_length])
    return text

def scrape_and_correlate(url, base_url="https://weaviate.io/"):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    max_length = 10
    image_text_pairs = []

    for img in soup.find_all('img'):
        image_url = img.get('src')
        image_url = urljoin(base_url, image_url)
        text = img.find_next('p').text if img.find_next('p') else img.get('alt', '')

        if not text:
            continue

        truncate_text = truncated_text(text, max_length)

        if image_url.endswith(".png") or image_url.endswith(".jpg"):
            image = load_image_from_url(image_url)
            inputs = clip_processor(text=[truncate_text], images=image, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            image_embed = outputs.image_embeds
            text_embed = outputs.text_embeds
            similarity = torch.nn.functional.cosine_similarity(image_embed, text_embed).item()
            image_text_pairs.append((image_url, text, similarity))

    image_text_pairs.sort(key=lambda x: x[2], reverse=True)
    return image_text_pairs

# Functions for Weaviate Operations
def create_weaviate_collection(name):
    client.collections.create(
        name=name,
        properties=[
            wc.Property(name="text", data_type=wc.DataType.TEXT),
            wc.Property(name="imageUrl", data_type=wc.DataType.TEXT),
            wc.Property(name="textContent", data_type=wc.DataType.TEXT),
            wc.Property(name="type", data_type=wc.DataType.TEXT),
            wc.Property(name="image", data_type=wc.DataType.BLOB),
        ],
        vectorizer_config=wc.Configure.Vectorizer.multi2vec_clip(
            image_fields=[wc.Multi2VecField(name="image", weight=0.9)],  # 90% of the vector is from the poster
            text_fields=[wc.Multi2VecField(name="text", weight=0.1)],  # 10% of the vector is from the title
        ),
        generative_config=wc.Configure.Generative.openai()
    )

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content).decode('utf-8')

def store_in_weaviate(collection, data):
    for item in data:
        data_object = {
            "textContent": item[1],
            "text": item[1],
            "type": "text"
        }
        collection.data.insert(properties=data_object)
        print(f"stored {item[1]} content")

        if item[0].endswith(".png") or item[0].endswith(".jpg"):
            blob_string = get_as_base64(item[0])
            data_object = {
                "type": "image",
                "image": blob_string,
                "imageUrl": item[0],
            }
            collection.data.insert(properties=data_object)
            print(f"stored {item[0]} content")

def vectorize_text(text):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    outputs = model.get_text_features(**inputs)
    return outputs.detach().numpy().flatten().tolist()

def query_data(query, collection):
    query_vector = vectorize_text(query)
    import weaviate.classes.query as wq
    response = collection.query.hybrid(
        query=query, limit=4, return_metadata=wq.MetadataQuery(score=True)
    )
    return response

def extract_data(url):
    name = "weaviate_new1"
    image_text_pairs = scrape_and_correlate(url)
    if name not in client.collections.list_all().keys():
        create_weaviate_collection(name)
    collection = client.collections.get(name)
    store_in_weaviate(collection, image_text_pairs)

def generate_context(data):
    text = []
    images = []

    for obj in data:
        if obj.properties['text'] is not None:
            text.append(obj.properties['text'])
        if obj.properties['imageUrl'] is not None:
            images.append(obj.properties['imageUrl'])
    return text, images

# Functions for Generative Operations
def generate_content(text, images, query):
    content = []
    text = "\n".join(text)
    content.append({"type": "text",
                    "text": f"Context: {text}", })
    content.append({"type": "text",
                    "text": f"Generate answer from given context and images. {query}. Explain in brief considering all context and in 7-8 lines only", })
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": img
        })
    return content

def generate_image_content(collection, prompt, text):
    image_descriptions = []

    import weaviate.classes.query as wq
    from weaviate.classes.query import Filter
    img_response = collection.query.hybrid(
        query=prompt, limit=3, return_metadata=wq.MetadataQuery(score=True),
        filters=Filter.by_property("type").equal("image"),
    )

    def describe_image(url, text):
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"Describe the image as per given below context {text}. Keep it brief and according to context query. 2 to 4 lines",
                },
                {
                    "type": "image_url",
                    "image_url": url
                },
            ]
        )

        output = ChatGoogleGenerativeAI(model="gemini-pro-vision").invoke([message])
        return output.content

    images = []
    for obj in img_response.objects:
        if obj.properties['imageUrl'] is not None:
            images.append(obj.properties['imageUrl'])

    for i in images:
        response = describe_image(i, text)
        image_descriptions.append(response)
    return images, image_descriptions

# Functions for Gradio Interface
def reset(model, stream, temperature, stop_sequence, top_k, top_p):
    model.value = "gemini-pro-vision"
    stream.value = True
    temperature.value = 0.6
    stop_sequence.value = ""
    top_k.value = 8
    top_p.value = 0.4

def gemini_generator_run(prompt, model, stream, temperature, stop_sequence, top_k, top_p):
    collection = client.collections.get("weaviate_new1")

    model_config = {
        "model": model,
        "stream": stream,
        "temperature": temperature,
        "stop_sequence": stop_sequence,
        "top_k": top_k,
        "top_p": top_p
    }

    response_objects = query_data(prompt, collection)
    text, images = generate_context(response_objects.objects)
    response_images, image_descriptions = generate_image_content(collection, prompt, text)
    content = generate_content(text, images, prompt)

    message = HumanMessage(content=content)
    response = ChatGoogleGenerativeAI(**model_config).invoke([message])

    final_response = []
    final_response.append(f"<p>{response.content}</p>")
    for img, desc in zip(response_images, image_descriptions):
        final_response.append(f"<img src='{img}'>")
        final_response.append(f"<p>{desc}</p>")

    return " ".join(final_response)

def query_message(chatbot, query, model, stream, temperature, stop_sequence, top_k, top_p):
    response = gemini_generator_run(query, model, stream, temperature, stop_sequence, top_k, top_p)
    chatbot.append((None, f"<html>{response}</html>"))
    return chatbot, ""

def integrate_gradio_interface():
    with gr.Blocks() as interface:
        gr.Markdown("<h2><center>Generative Ai model using Google Gemini Pro Vision</center></h2>")
        gr.Markdown("<center><h3><a href='https://geminiprovision.com/' target='_blank'>Visit More Information about Google Gemini Pro Vision Here</a></h3></center>")

        with gr.Row():
            with gr.Column(scale=0.70):
                chatbot = gr.Chatbot()
                with gr.Row():
                    query = gr.Textbox(show_label=False, placeholder="Enter text here.").style(container=False)
                    submit = gr.Button("Submit", variant="primary")
                with gr.Accordion("Parameters", open=False):
                    model = gr.Textbox(label="Model", value="gemini-pro-vision")
                    stream = gr.Checkbox(label="Stream", value=True)
                    temperature = gr.Slider(minimum=0, maximum=1, value=0.6, label="Temperature")
                    stop_sequence = gr.Textbox(label="Stop sequence", value="")
                    top_k = gr.Slider(minimum=1, maximum=20, value=8, label="Top-k")
                    top_p = gr.Slider(minimum=0, maximum=1, value=0.4, label="Top-p")
                    submit_parameters = gr.Button("Submit Parameters")
                    submit_parameters.click(
                        fn=reset,
                        inputs=[model, stream, temperature, stop_sequence, top_k, top_p],
                        outputs=[model, stream, temperature, stop_sequence, top_k, top_p]
                    )
            with gr.Column(scale=0.30):
                gr.Examples(
                    examples=[
                        "Generate a response on given text",
                        "Generate an image",
                        "Generate a response on both text and image",
                        "Generate a response on both text and image with image content",
                    ],
                    inputs=query
                )
                gr.Markdown("Use any of the above examples or enter your own query to interact with the model.")
            submit.click(
                fn=query_message,
                inputs=[chatbot, query, model, stream, temperature, stop_sequence, top_k, top_p],
                outputs=[chatbot, query],
                queue=False
            )

    interface.launch()

# Main Execution
if __name__ == "__main__":
    integrate_gradio_interface()
