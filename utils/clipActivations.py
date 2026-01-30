import numpy as np
import matplotlib.pyplot as plt
import torch
import clip

class Clip():
    def __init__(self, model, concept_file, device, class_list):
        self.model = model
        self.concept_file = concept_file
        self.device = device
        self.class_list = class_list

    # Modify to read concepts from json iteratively
    def collect_concepts(self):
        with open(self.concept_file, "r") as file:
            self.concepts = file.read().split("\n")
    
    def collect_embeddings(self):
        clip_model, processes = clip.load(self.model, device=self.device)
        text_inputs = clip.tokenize(self.concepts)

        with torch.no_grad():
            text_inputs = text_inputs.to(self.device)
            text_features = clip_model.encode_text(text_inputs)

        text_features /= text_features.norm(dim=-1, keepdim=True)
        print(text_features.shape)
    
    # Create file paths for the embedding files
    def save_embeddings(self):
        pass
