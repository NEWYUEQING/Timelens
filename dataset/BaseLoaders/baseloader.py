import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class BaseLoader(Dataset):
    """Minimal loader for frame-aligned RGB/event interpolation data.

    Assumption for this simplified branch:
    - Event files are frame-aligned with RGB files (same timeline index).
    - Training/inference only needs intermediate timestamp event maps.
    """

    def __init__(self, para, training=True):
        self.para = para
        self.training_flag = training
        self.key = 'training_config' if self.training_flag else 'validation_config'

        self.crop_size = para[self.key]['crop_size']
        self.data_paths = para[self.key]['data_paths']
        self.rgb_sampling_ratio = para[self.key]['rgb_sampling_ratio']
        self.interp_ratio = para[self.key]['interp_ratio']
        self.sample_group = (
            self.interp_ratio - 1
            if 'sample_group' not in para[self.key].keys()
            else para[self.key]['sample_group']
        )
        self.random_t = para[self.key]['random_t']
        self.color = 'gray' if 'color' not in para[self.key].keys() else para[self.key]['color']

        self.totensor = ToTensor()
        self.samples_list = []
        self.samples_indexing()

        if len(self.samples_list) > 0:
            _, _, _, evs_sample = self.samples_list[0]
            print(f'Training: {training}, EVS len {len(evs_sample)} (frame-aligned mode)')

    def samples_indexing(self):
        self.samples_list = []
        for folder_key in self.data_paths.keys():
            rgb_path, evs_path = self.data_paths[folder_key]

            # simplified branch: strictly frame-aligned event maps
            if len(rgb_path) != len(evs_path):
                raise ValueError(
                    f'Frame-aligned mode requires len(rgb_path)==len(evs_path), '
                    f'got {len(rgb_path)} vs {len(evs_path)} for {folder_key}'
                )

            indexes = list(range(0, len(rgb_path), self.rgb_sampling_ratio))
            step = 1 if self.training_flag else self.interp_ratio
            for i_ind in range(0, len(indexes) - self.interp_ratio, step):
                rgb_inds = indexes[i_ind:i_ind + self.interp_ratio + 1]
                rgb_sample = [rgb_path[sind] for sind in rgb_inds]
                evs_sample = [evs_path[sind] for sind in rgb_inds]
                rgb_name = [os.path.splitext(os.path.split(rs)[-1])[0] for rs in rgb_sample]
                self.samples_list.append([folder_key, rgb_name, rgb_sample, evs_sample])

    def __len__(self):
        return len(self.samples_list)

    def imreader(self, impath):
        im = self.totensor(Image.open(impath))
        if self.color == 'gray':
            im = im[0] * 0.299 + im[1] * 0.587 + im[2] * 0.114
            im = im.unsqueeze(0)
        return im

    def ereader(self, events_path):
        event_tensors = []
        for epath in events_path:
            edata = np.load(epath, allow_pickle=True)['data']
            if len(edata.shape) == 2:
                edata = np.expand_dims(edata, 0)
            event_tensors.append(torch.from_numpy(edata))
        return torch.cat(event_tensors, 0).float()

    def data_loading(self, paths, sample_t):
        folder_name, rgb_name, rgb_sample, evs_sample = paths
        im0 = self.imreader(rgb_sample[0])
        im1 = self.imreader(rgb_sample[-1])
        events = self.ereader([evs_sample[st] for st in sample_t])
        gts = [self.imreader(rgb_sample[st]) for st in sample_t]
        return folder_name, rgb_name, im0, im1, events, gts

    def __getitem__(self, item):
        item_content = self.samples_list[item]
        if self.random_t:
            sample_t = random.sample(range(1, self.interp_ratio // 2), self.sample_group // 2)
            sample_t.append(self.interp_ratio // 2)
            sample_t.extend(
                random.sample(
                    range(self.interp_ratio // 2 + 1, self.interp_ratio),
                    self.sample_group // 2,
                )
            )
        else:
            sample_t = list(range(1, self.interp_ratio))

        folder_name, rgb_name, im0, im1, events, gts = self.data_loading(item_content, sample_t)
        h, w = im0.shape[1:]
        if self.crop_size:
            hs, ws = random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)
            im0 = im0[:, hs:hs + self.crop_size, ws:ws + self.crop_size]
            im1 = im1[:, hs:hs + self.crop_size, ws:ws + self.crop_size]
            events = events[:, hs:hs + self.crop_size, ws:ws + self.crop_size]
            gts = [gt[:, hs:hs + self.crop_size, ws:ws + self.crop_size] for gt in gts]
        else:
            hn, wn = h // 32 * 32, w // 32 * 32
            im0, im1, events = im0[:, :hn, :wn], im1[:, :hn, :wn], events[:, :hn, :wn]
            gts = [gt[:, :hn, :wn] for gt in gts]

        gts = torch.cat(gts, 0)
        left_weight = [1 - float(st) / self.interp_ratio for st in sample_t]
        rgb_name = [os.path.splitext(r)[0] for r in rgb_name]
        return {
            'folder': folder_name,
            'rgb_name': [rgb_name[0]] + [rgb_name[st] for st in sample_t] + [rgb_name[-1]],
            'im0': im0,
            'im1': im1,
            'gts': gts,
            'events': events,
            't_list': sample_t,
            'left_weight': left_weight,
            'interp_ratio': self.interp_ratio,
        }
