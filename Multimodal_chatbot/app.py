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

import warnings
warnings.filterwarnings('ignore')

# Configure Google API Key
os.environ['GOOGLE_API_KEY'] = "API_KEY"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Initialize CLIP model
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Connect to Weaviate
client = weaviate.connect_to_local()
chatbot = []

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
            image_fields=[wc.Multi2VecField(name="image", weight=0.9)],
            text_fields=[wc.Multi2VecField(name="text", weight=0.1)],
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
        print(f"Stored {item[1]} content")

        if item[0].endswith(".png") or item[0].endswith(".jpg"):
            blob_string = get_as_base64(item[0])
            data_object = {
                "type": "image",
                "image": blob_string,
                "imageUrl": item[0],
            }
            collection.data.insert(properties=data_object)
            print(f"Stored {item[0]} content")
    return "Data extracted successfully"

def vectorize_text(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    outputs = clip_model.get_text_features(**inputs)
    return outputs.detach().numpy().flatten().tolist()

def query_data(query, collection):
    query_vector = vectorize_text(query)
    import weaviate.classes.query as wq
    from weaviate.classes.query import Filter

    response = collection.query.hybrid(
        query=query, limit=5, return_metadata=wq.MetadataQuery(score=True),
        filters=Filter.by_property("type").equal("image"),
    )
    return response.objects

def extract_data(url):
    client.collections.delete("Data1")
    name = "Data1"
    image_text_pairs = scrape_and_correlate(url)
    if name not in client.collections.list_all().keys():
        create_weaviate_collection(name)
    collection = client.collections.get(name)
    status = store_in_weaviate(collection, image_text_pairs)
    return status

def generate_context(data):
    text = []
    images = []

    for obj in data:
        if obj.properties['text'] != None:
            text.append(obj.properties['text'])
        if obj.properties['imageUrl'] != None:
            images.append(obj.properties['imageUrl'])
    return text, images

def generate_content(text, images, query):
    content = []
    text = "/n".join(text)
    content.append({"type": "text", "text": f"Context: {text}"})
    if "give image" in query:
        content.append({"type": "text", "text": f"Generate an image based on the query."})
    else:
        content.append({"type": "text", "text": f"Generate answer from the context and images. {query}"})
    for img in images:
        content.append({"type": "image_url", "image_url": img})
    return content

def generate_image_content(collection, prompt, text, file):
    import weaviate.classes.query as wq
    from weaviate.classes.query import Filter
    from pathlib import Path

    # Fetch images based on the query
    if file:
        img_response = collection.query.near_image(
            near_image=Path(file),
            limit=2, return_metadata=wq.MetadataQuery(score=True),
        )
    else:
        img_response = collection.query.hybrid(
            query=prompt, limit=2, return_metadata=wq.MetadataQuery(score=True),
            filters=Filter.by_property("type").equal("image"),
        )

    # Function to generate a description for an image
    def describe_image(url, text):
        message = HumanMessage(
            content=[{"type": "text", "text": f"Describe the image as per the given context {text}. Keep it brief."},
                     {"type": "image_url", "image_url": url}]
        )
        output = ChatGoogleGenerativeAI(model="gemini-1.5-flash").invoke([message])
        return output

    # Prepare the final content with image and description
    images = [obj.properties['imageUrl'] for obj in img_response.objects]
    image_descriptions = [describe_image(i, text) for i in images]

    # Pair the image with its description
    content = []
    for img_url, description in zip(images, image_descriptions):
        content.append({"type": "text", "text": description.content})  # Add description text
        content.append({"type": "image_url", "image_url": img_url})  # Add the image

    return content


def gemini_generator_run(prompt, model, stream, temperature, stop_sequence, top_k, top_p, file):
    GlobantWebData_Coll = client.collections.get("Data1")
    model_config = {
        "model": model,
        "stream": stream,
        "temperature": temperature,
        "stop_sequence": stop_sequence,
        "top_k": top_k,
        "top_p": top_p
    }
    response_objects = query_data(prompt, GlobantWebData_Coll)
    text, images = generate_context(response_objects)
    response_images, image_descriptions,extra_value = generate_image_content(GlobantWebData_Coll, prompt, text, file)
    content = generate_content(text, images, prompt)
    
    message = HumanMessage(content=content)
    response = ChatGoogleGenerativeAI(**model_config).invoke([message])

    final_response = [response.content]
    for i, j in zip(response_images, image_descriptions):
        final_response.append(f"Image URL: {i}")
        final_response.append(f"Description: {j}")

    return "\n".join(final_response)

def query_message(chatbot, url, query, model, stream, temperature, stop_sequence, top_k, top_p, file):
    response = gemini_generator_run(query, model, stream, temperature, stop_sequence, top_k, top_p, file)
    chatbot.append((None, response))
    return chatbot

def reset():
    return "gemini-1.5-flash", True, 0.6, "", 8, 0.4

# Main function to integrate Gradio Interface
def integrate_gradio_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=3):
                # Chatbot section
                gr.Markdown("### Chatbot")
                chatbot = gr.Chatbot(show_copy_button=True, height=500)
                chatbot.render_css = "overflow-y: scroll; height: 400px;"
                
                # User input section: query and file upload
                with gr.Row():
                    with gr.Column(scale=6):
                        prompt = gr.Textbox(placeholder="Write query", label="Enter your query")
                    with gr.Column(scale=2):
                        file = gr.File(label="Upload File")
                    with gr.Column(scale=2):
                        button = gr.Button(value="Generate Answer")
            
            with gr.Column(scale=1):
                # URL input and extraction
                with gr.Column(scale=6):
                    url_input = gr.Textbox(label="URL", placeholder="Enter URL")
                with gr.Column(scale=1):
                    button_extract = gr.Button(value="Extract Information")
                
                # Status message output
                with gr.Row():
                    status_message = gr.Label(label="Status")
                
                # Model parameters section
                gr.Markdown("### Model Parameters")
                reset_params = gr.Button(value="Reset")
                model = gr.Dropdown(value="gemini-1.5-flash", choices=["gemini-pro", "gemini-1.5-flash"],
                                    label="Model", interactive=True)
                stream = gr.Radio(label="Streaming", choices=[True, False], value=True, interactive=True)
                temperature = gr.Slider(value=0.6, minimum=0.1, maximum=1.0, label="Temperature", interactive=True)
                stop_sequence = gr.Textbox(label="Stop Sequence")
                
                # Advanced settings
                gr.Markdown("### Advanced Settings")
                top_k = gr.Slider(value=8, minimum=1, maximum=100, label="Top-k", interactive=True)
                top_p = gr.Slider(value=0.4, minimum=0.1, maximum=1.0, label="Top-p", interactive=True)

        # Button actions
        button.click(query_message, 
                     inputs=[chatbot, url_input, prompt, model, stream, temperature, stop_sequence, top_k, top_p, file], 
                     outputs=chatbot)
        
        button_extract.click(fn=extract_data, inputs=[url_input], outputs=[status_message])
        
        reset_params.click(fn=reset, 
                           inputs=[model, stream, temperature, stop_sequence, top_k, top_p], 
                           outputs=[model, stream, temperature, stop_sequence, top_k, top_p])

        demo.launch()

# Main execution
if __name__ == "__main__":
    integrate_gradio_interface()
