from transformers import Owlv2Processor, Owlv2ForObjectDetection

import cv2
import skimage
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt

def plot_predictions(input_image, text_queries, scores, boxes, labels, score_threshold):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, extent=(0, 1.0, 1.0, 0))
    ax.set_axis_off()

    for score, box, label in zip(scores, boxes, labels):
      if score < score_threshold:
        continue
      else:
        print(f'label: {text_queries[label]}     score: {score}    box pose: {box}')
      cx, cy, w, h = box
      
      ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
              [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
      ax.text(
          cx - w / 2,
          cy + h / 2 + 0.015,
          f"{text_queries[label]}: {score:1.2f}",
          ha="left",
          va="top",
          color="red",
          bbox={
              "facecolor": "white",
              "edgecolor": "red",
              "boxstyle": "square,pad=.3"
          })
    plt.savefig('/home/pjw971022/Sembot/real_bot/save_vision/obs/detected_obs.png')


# Download sample image
image_path = '/home/pjw971022/Sembot/real_bot/save_vision/obs/rgb_obs.png'  #skimage.data.astronaut()
image = Image.open(image_path).convert("RGB")
# image = skimage.data.astronaut()
# image = Image.fromarray(np.uint8(image)).convert("RGB")
import torch

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Process image and text inputs
# Text queries to search the image for
# text_queries1 = ["human face", "rocket", "nasa badge", "star-spangled banner"]
text_queries = ["baseball", "ball", "cup", "fan", "dice", "tape", "ring", "phone", "controller", "headset", "tool"]
# text_queries = [["a photo of a cat", "a photo of a dog"]]

processor1 = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble", do_pad = False)
model1 = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

model1 = model1.to(device)
model1.eval()
inputs1 = processor1(text=text_queries, images=image, return_tensors="pt",).to(device)
# print("pixel val1: ", inputs1['pixel_values'][-1,0,-1])


from transformers import OwlViTProcessor, OwlViTForObjectDetection
# model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
processor2 = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

inputs2 = processor2(text=text_queries, images=image, return_tensors="pt").to(device)
# print("pixel val2: ", inputs2['pixel_values'][-1,0,-1])

# # Get predictions
# with torch.no_grad():
#   outputs1 = model(**inputs1)

# inputs = processor(text=text_queries2, images=image, return_tensors="pt").to(device)

# print("pixel2: ", inputs['pixel_values'])

# Get predictions
with torch.no_grad():
  outputs = model1(**inputs1)




# # Threshold to eliminate low probability predictions
score_threshold = 0.3


import matplotlib.pyplot as plt

from transformers.image_utils import ImageFeatureExtractionMixin
mixin = ImageFeatureExtractionMixin()

# # Load example image
image_size = model1.config.vision_config.image_size
image = mixin.resize(image, image_size)
input_image = np.asarray(image).astype(np.float32) / 255.0

target_sizes = torch.Tensor([image.size[::-1]])
print(target_sizes)

# Get prediction logits
logits = torch.max(outputs["logits"][0], dim=-1)
scores = torch.sigmoid(logits.values).cpu().detach().numpy()

# Get prediction labels and boundary boxes
labels = logits.indices.cpu().detach().numpy()
boxes = outputs["pred_boxes"][0].cpu().detach().numpy()
results = processor1.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=score_threshold)

# print(f"box: {results[0]['boxes']}")


plot_predictions(input_image, text_queries, scores, boxes, labels, score_threshold)