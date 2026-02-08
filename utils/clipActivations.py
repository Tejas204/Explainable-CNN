import numpy as np
import matplotlib.pyplot as plt
import torch
import clip
import json
import safetensors
import math
import random
import os
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

    # Create permutations for training
    def create_permutations(self, filename, n_permutations):
        print(f"Generating {n_permutations} permutations for {filename}")
        concept_embeddings = {}
        with safe_open(f'embeddings/{filename}.safetensors', framework='pt', device=0) as file:
            for k in file.keys():
                concept_embeddings[k] = file.get_tensor(k)
        
        
        concept_keys = list(concept_embeddings.keys())

        concept_permutations = {}
        for i in range(n_permutations):
            concept_permutations[i] = random.sample(concept_keys, len(concept_keys))

        # concept_embedding_permutations = {}
        # for i in range(len(concept_permutations)):
        #     for j in range(len(concept_permutations[i])):
        #         concept_embedding_permutations[f"{i}_{concept_permutations[i][j]}"] = concept_embeddings[concept_permutations[i][j]]
        
        os.makedirs("permutations", exist_ok=True)
        with open(f'permutations/{filename}.json', 'w') as f:
            json.dump(concept_permutations, f, indent=4)
        print(f"Saved json file for permutations of {filename} successfully!")