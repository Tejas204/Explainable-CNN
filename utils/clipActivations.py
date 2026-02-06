import numpy as np
import matplotlib.pyplot as plt
import torch
import clip
import json
import safetensors
import math
from safetensors.torch import save_file
from safetensors import safe_open
from itertools import permutations

class Clip():
    def __init__(self, model, concept_file, device, class_list):
        self.model = model
        self.concept_file = concept_file
        self.device = device
        self.class_list = class_list
        self.class_concept_embeddings = {}

    # Modify to read concepts from json iteratively
    def collect_concepts(self):
        with open(self.concept_file, "r") as file:
            self.concepts = json.load(file)
    
    def collect_embeddings(self):
        clip_model, processes = clip.load(self.model, device=self.device)

        for label in self.concepts.keys():
            text_inputs = clip.tokenize(self.concepts[label])

            with torch.no_grad():
                text_inputs = text_inputs.to(self.device)
                text_features = clip_model.encode_text(text_inputs)

            text_features /= text_features.norm(dim=-1, keepdim=True)
            self.class_concept_embeddings[label] = text_features
    
    # Create file paths for the embedding files
    def save_embeddings(self):
        for key in self.concepts.keys():
            embedding = {}
            for i in range(len(self.concepts[key])):
                embedding[self.concepts[key][i]] = self.class_concept_embeddings[key][i]
            save_file(embedding, "embeddings/"+str(key)+".safetensors")
            print(f"Saved safetensor file for {key} successfully!")

    def create_permutations(self):
        concept_permutations = []
        with safe_open('embeddings/airplane.safetensors', framework='pt', device=0) as file:
            concepts = file.keys()

            print(f"Total Permutations: {math.factorial(len(concepts))}")
            permuts = list(permutations(['air traffic tower', 'aircraft livery', 'airline logo']))
            
            for permut in permuts:
                concept_embeddings = {}
                for i in range(len(permut)):
                    concept_embeddings[permut[i]] = file.get_tensor(permut[i])
                concept_permutations.append(concept_embeddings)

            print(concept_permutations)