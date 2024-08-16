# Creator Cui Liz
# Time 20/06/2024 17:08
import os
import csv
import re
import time

import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import yaml
from typing import List

csv.field_size_limit(1000000)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SpamIsHamDataset(Dataset):

    def __init__(self, max_tokens: int = 512):
        self.tokenizer = BertTokenizer.from_pretrained("/root/SpamProj/Dataset/bert-base-uncased")
        self.model = BertModel.from_pretrained("/root/SpamProj/Dataset/bert-base-uncased").to(DEVICE).eval()
        # Get the dataset path
        self.dataset_folder = "/root/SpamProj/Dataset"
        self.dataset_name: list[str] = sorted(i for i in os.listdir(self.dataset_folder) if i[-4:] == ".csv")

        self.paths = paths
        self.max_tokens = max_tokens

        # Data has been cleaned
        self.label_list = []
        self.content_list = []
        self.load_data()

    def load_data(self):
        for path in self.paths:
            with open(path, mode="r") as f:
                reader = csv.reader(f)
                next(reader)
                n_hams = 0
                n_spams = 0
                for row in tqdm(reader, desc=f"Loading {path}"):
                    if row:
                        lower_content = "".join(row[1:]).lower()
                        if lower_content == "":
                            continue
                        self.content_list.append(" ".join(lower_content.split(" ")[:self.max_tokens]))
                        self.label_list.append(int(row[0][0].lower() in ["s", "1"]))
                        if self.label_list[-1] == 1:
                            n_spams += 1
                        else:
                            n_hams += 1
                print(f"{n_hams=}, {n_spams=}")

    def to_tensor(self, content: str) -> torch.Tensor:
        if isinstance(content, str):
            encode_input = self.tokenizer.encode_plus(content, padding="max_length", truncation=True,
                                                      max_length=self.max_tokens, return_tensors="pt")
        else:
            encode_input = self.tokenizer.batch_encode_plus(content, padding="max_length", truncation=True,
                                                            max_length=self.max_tokens, return_tensors="pt")

        ids = encode_input["input_ids"].to(DEVICE)
        mask = encode_input["attention_mask"].to(DEVICE)
        with torch.no_grad():
            outputs = self.model(ids, attention_mask=mask)

        return outputs.last_hidden_state

    def __len__(self):
        return len(self.content_list)

    def __getitem__(self, slice_idx):
        if isinstance(slice_idx, list):
            content = [self.content_list[each] for each in slice_idx]
            label = [self.label_list[each] for each in slice_idx]
        else:
            content = self.content_list[slice_idx]
            label = self.label_list[slice_idx]
        sentence_embed = self.to_tensor(content)

        return sentence_embed, torch.tensor(label, dtype=torch.long, device=DEVICE).view(-1, 1)


def collate_fn(batch):
    sentence_embeds, labels = zip(*batch)
    return torch.cat(sentence_embeds, dim=0), torch.cat(labels, dim=0)


def createDatasetCache(dataset, chunk_size: int, prefix: str):
    all_embeds = torch.zeros(chunk_size, 512, 768, device=DEVICE, dtype=torch.float32)
    all_labels = torch.zeros(chunk_size, 1, device=DEVICE, dtype=torch.long)

    data_count = len(dataset)
    pbar = tqdm(range(data_count), desc="Making Dataset Cache")
    shuffle_indices = list(range(len(dataset)))
    random.shuffle(shuffle_indices)
    for i in pbar:
        data_idx = shuffle_indices[i]
        sentence_embed, label = dataset[data_idx]

        all_embeds[i % chunk_size] = sentence_embed
        all_labels[i % chunk_size] = label

        if i != 0 and i % chunk_size == 0:
            chunk_num = i // chunk_size
            print(f"Saving Chunk {chunk_num}")
            torch.save([all_embeds, all_labels], f"{prefix}_chunk_{chunk_num}_{chunk_size}.pth")  # chunk_num

    # Save last chunk
    chunk_num = data_count // chunk_size + 1
    remaining_num = data_count % chunk_size
    print(f"Saving Chunk {chunk_num}, last chunk number of items = {remaining_num}")
    torch.save([all_embeds[:remaining_num], all_labels[:remaining_num]],
               f"{prefix}_chunk_{chunk_num}_{remaining_num}.pth")
    print("Done!")


class CacheDataset(Dataset):
    def __init__(self, chunk_paths: List[str], raw_in_each_chunk: int, gpt_load_percentage: float,
                 load_all: bool = False):
        self.chunk_paths = chunk_paths
        self.num_chunks = len(chunk_paths)
        self.current_chunk = 0
        self.load_all = load_all
        self.all_embeds = []
        self.all_labels = []

        full_chunk_size = int(self.chunk_paths[0].split("_")[-1][:-4])
        gpt_in_each_chunk = (full_chunk_size - raw_in_each_chunk)
        self.chunk_clip_size = raw_in_each_chunk + int(gpt_in_each_chunk * gpt_load_percentage)

        self.data_count = self.chunk_clip_size * self.num_chunks

        if self.load_all:
            for i in range(self.num_chunks):
                embeds, labels = self.loadNextChunk()
                self.all_embeds.extend(embeds)
                self.all_labels.extend(labels)

    def loadNextChunk(self):
        self.current_chunk = (self.current_chunk + 1) % len(self.chunk_paths)
        dataset = torch.load(self.chunk_paths[self.current_chunk], map_location="cpu")
        embeds = list(dataset[0].split(1, 0))[:self.chunk_clip_size]
        labels = list(dataset[1].split(1, 0))[:self.chunk_clip_size]
        return embeds, [each.to(torch.float32) for each in labels]

    def __len__(self):
        return self.data_count

    def __getitem__(self, idx):
        if self.load_all:
            return self.all_embeds[idx].to(DEVICE), self.all_labels[idx].to(DEVICE)
        else:
            if len(self.all_labels) == 0:
                self.all_embeds, self.all_labels = self.loadNextChunk()
            random_idx = random.randint(0, len(self.all_labels) - 1)
            embed = self.all_embeds.pop(random_idx).to(DEVICE)
            label = self.all_labels.pop(random_idx).to(DEVICE)
            return embed, label.to(torch.float32)


if __name__ == '__main__':
    set_name = "test"

    with open("DatasetPaths.yaml", "r") as in_file:
        dataset = SpamIsHamDataset(paths=yaml.safe_load(in_file)[f"{set_name}_paths"])

    createDatasetCache(dataset, chunk_size=10000, prefix=f"/root/autodl-tmp/test_chunks/{train}")