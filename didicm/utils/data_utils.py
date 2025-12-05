import torch
import timm
from collections import defaultdict


def change_dataset_to_subset_for_data_ratio(dataset, data_ratio, seed=42):
    assert type(dataset) == timm.data.dataset.ImageDataset, "Dataset must be a timm.data.dataset.ImageDataset"
    assert data_ratio > 0 and data_ratio <= 1, "Data ratio must be between 0 and 1"

    if data_ratio < 1.0:
        class_to_samples = defaultdict(list)
        for sample in dataset.reader.samples:
            class_to_samples[sample[1]].append(sample)
        for i in range(len(class_to_samples)):
            samples = class_to_samples[i]
            rand_idx = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed))[:int(len(samples) * data_ratio)]
            class_to_samples[i] = [samples[j] for j in rand_idx]
        samples = [sample for samples in class_to_samples.values() for sample in samples]
        dataset.reader.samples = samples
    
    return dataset
