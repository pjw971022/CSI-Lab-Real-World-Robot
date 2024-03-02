import requests
from PIL import Image
import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection
import matplotlib.pyplot as plt

class OWLViTDetector:
    def __init__(self, device, score_threshold):
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble", do_pad = False)
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.device = device
        self.model = model.to(device)
        self.score_threshold = score_threshold

    def forward(self, image, categories):
        inputs = self.processor(text=categories, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get prediction logits
        logits = torch.max(outputs["logits"][0], dim=-1)
        scores = torch.sigmoid(logits.values).cpu().detach().numpy()

        # Get prediction labels and boundary boxes
        labels = logits.indices.cpu().detach().numpy()
        boxes = outputs["pred_boxes"][0].cpu().detach().numpy()

        return {'boxes':boxes, 'scores': scores, 'labels': labels}

    def plot_predictions(self, input_image, text_queries, outputs, image_path):
        scores= outputs['scores']
        boxes = outputs['boxes']
        labels = outputs['labels']
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(input_image, extent=(0, 1, 0.75, 0))
        ax.set_axis_off()
        ax.plot(0., 0.0,'ro')

        for score, box, label in zip(scores, boxes, labels):
            if score < self.score_threshold:
                continue

            cx, cy, w, h = box
            cy *= 0.75
            h *= 0.75
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
        plt.savefig(image_path)
    # plot_predictions(input_image, text_queries, scores, boxes, labels)