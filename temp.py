import io
import base64
import numpy as np
import torch
from PIL import Image

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


def image_to_base64_data_uri(image_input):
    # Check if the input is a file path (string)
    if isinstance(image_input, str):
        with open(image_input, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')

    # Check if the input is a PIL Image
    elif isinstance(image_input, Image.Image):
        buffer = io.BytesIO()
        image_input.save(buffer, format="PNG")  # You can change the format if needed
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    else:
        raise ValueError("Unsupported input type. Input must be a file path or a PIL.Image.Image instance.")

    return f"data:image/png;base64,{base64_data}"



mmproj="/home/pjw971022/Sembot/VideoRAG/mmproj-model-f16.gguf"
model_path="/home/pjw971022/Sembot/VideoRAG/ggml-model-q4_0.gguf"

chat_handler = Llava15ChatHandler(clip_model_path=mmproj, verbose=False)
spacellava = Llama(model_path=model_path, chat_handler=chat_handler, n_ctx=2048, logits_all=True, n_gpu_layers = -1, verbose=False)


# @title Set your image and prompt
image_path = '/home/pjw971022/Sembot/VideoRAG/warehouse_test.jpg' # @param {type:"string"}
prompt = "how high is the stack of boxes on the pallet held up by the forklift?" # @param {type:"string"
# prompt = "Tell me what the object at the front among the items on the desk is. There is several objects on it, including a bookshelf, a spam can, a mustard, domino sugar, and a crackers."
# prompt = "Please estimate the distance from the robotic arm to each object on the table. The distance to the table is 90cm. There is several objects on it, including a bookshelf, a spam can, a mustard, domino sugar, and a crackers." # @param {type:"string"}

if '.' in image_path.split('/')[-1]:
    # Load the image and convert it to base64 data uri
    image_path = image_path 
    data_uri = image_to_base64_data_uri(image_path)
    system_query = """You are participating in a visual question answering game with your
    friend. In this game, you are presented with a question which requires visual information from an
    image to answer. You can see the question but not the image, while your friend 1 can see the image but
    not the original question. Luckily, you are allowed to decompose the question and ask your friend
    about the image. Your friend gives you answers which can be used to answer the original question.
    """
    CoT_example = """
    Here is a sample conversation:
    [Question] How can I clean up the table? Give detailed instruction about how should I move my hand.
    [You] What objects are there in the image?
    [Friend] There is an empty coke can, a trash bin and a coffee machine.
    [You] Is the trash bin to the left or to the right of the coke can?
    [Friend] It's to the left.
    [You] Is the trash bin or the coke can further from you?
    [Friend] They are similar in depth.
    [You] How much to the left is the trash bin compared to the coke can?
    [Friend] Around 20 centimeters.
    [Answer] One should grab the coke can, move it 20 centimeters left and release it so it falls in the trash bin.
    Here is another example:
    [Question] Tell me if the distance
    between the blue bottle and the yellow book is longer than that between the plant and the coke can?
    [You] What is the distance between the blue bottle and the yellow book?
    [Tool] 0.3m
    [You] What is the distance between the plant and the coke can?
    [Friend] 0.7m
    [Robot] Since the distance between the blue bottle and the
    yellow book is 0.3m and distance between the plant while the coke can is 0.7m, the distance between
    the blue bottle and the yellow book is not longer than that between the plant and the coke can.
    [Answer] No.
    Here is another example:
    [Question] Which object can be reached by kids more easily, the white and yellow rabbit toy can or the dark green can of beer?
    [You] What is the elevation of the white and yellow rabbit toy can?
    [Friend] 0.9 m.
    [You] What is the elevation of the dark green can of beer?
    [Friend] 0.2 m.
    [Answer] Since the kids are generally shorter, it is easier for them to reach something that are lower in altitude, so it would be easier for them to reach the can of beer.
    Now, given a new question, try to answer the questions by asking your friend for related visual information.
    [Question]
    """
    messages = [
        {"role": "system", "content": system_query},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type" : "text", "text": prompt}
                ]
            }
        ]
    results = spacellava.create_chat_completion(messages = messages, temperature=0.2)
    file_name = image_path.split('/')[-1]
    print(f"file: {file_name}",results["choices"][0]["message"]["content"])
