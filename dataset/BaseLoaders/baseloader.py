import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class BaseLoader(Dataset):
    """Generic loader for RGB-event interpolation samples.

    Event alignment modes:
    - frame: events are timestamp-aligned to RGB frames (same count as RGB).
    - interval: events represent intervals between adjacent RGB frames (one fewer).
    - auto: infer per sequence by length comparison.
    """

    def __init__(self, para, training=True):
        self.para = para
        self.training_flag = training
        self.key = 'training_config' if self.training_flag else 'validation_config'
        self.crop_size = para[self.key]['crop_size']
        self.data_paths = para[self.key]['data_paths']
        self.data_index_offset = para[self.key]['data_index_offset']
        self.rgb_sampling_ratio = para[self.key]['rgb_sampling_ratio']
        self.interp_ratio = para[self.key]['interp_ratio']
        self.sample_group = (
            self.interp_ratio - 1
            if 'sample_group' not in para[self.key].keys()
            else para[self.key]['sample_group']
        )
        self.random_t = para[self.key]['random_t']
        self.event_mode = 'auto' if 'event_mode' not in para[self.key].keys() else para[self.key]['event_mode']
        self.events_channel = 128 if 'events_channel' not in para.keys() else para.events_channel

        self.totensor = ToTensor()
        self.samples_indexing()
        self.color = 'gray' if 'color' not in para[self.key].keys() else para[self.key]['color']

        if len(self.samples_list) > 0:
            _, _, _, evs_sample, sample_mode = self.samples_list[0]
            print(f'Training: {training}, EVS len {len(evs_sample)}, Event mode {sample_mode}')

    def _resolve_event_mode(self, rgb_len, evs_len):
        if self.event_mode in ('frame', 'interval'):
            return self.event_mode
        if evs_len == rgb_len:
            return 'frame'
        if evs_len == rgb_len - 1:
            return 'interval'
        raise ValueError(
            f'Cannot infer event mode with rgb_len={rgb_len}, evs_len={evs_len}. '
            f'Set {self.key}.event_mode to frame/interval explicitly.'
        )

    def samples_indexing(self):
        self.samples_list = []
        for k in self.data_paths.keys():
            rgb_path, evs_path = self.data_paths[k]
            event_mode = self._resolve_event_mode(len(rgb_path), len(evs_path))
            indexes = list(range(0, len(rgb_path), self.rgb_sampling_ratio))

            for i_ind in range(
                0,
                len(indexes) - self.interp_ratio,
                1 if self.training_flag else self.interp_ratio,
            ):
                rgb_inds = indexes[i_ind:i_ind + self.interp_ratio + 1]
                rgb_sample = [rgb_path[sind] for sind in rgb_inds]

                if event_mode == 'frame':
                    evs_inds = rgb_inds
                else:
                    evs_inds = indexes[i_ind:i_ind + self.interp_ratio]

                if max(evs_inds) >= len(evs_path):
                    continue

                evs_sample = [evs_path[sind] for sind in evs_inds]
                rgb_name = [os.path.splitext(os.path.split(rs)[-1])[0] for rs in rgb_sample]
                self.samples_list.append([k, rgb_name, rgb_sample, evs_sample, event_mode])
        return

    def __len__(self):
        return len(self.samples_list)

    def imreader(self, impath):
        im = self.totensor(Image.open(impath))
        if self.color == 'gray':
            im = im[0] * 0.299 + im[1] * 0.587 + im[2] * 0.114
            im = im.unsqueeze(0)
        return im

    def ereader(self, events_path):
        evs_data = []
        for epath in events_path:
            edata = np.load(epath, allow_pickle=True)['data']
            if len(edata.shape) == 2:
                edata = np.expand_dims(edata, 0)
            evs_data.append(torch.from_numpy(edata))

        evs_data = torch.cat(evs_data, 0).float()

        # Optional compatibility: fold to fixed channels when configured.
        if self.events_channel and evs_data.shape[0] > self.events_channel:
            fold_factor = evs_data.shape[0] // self.events_channel
            if evs_data.shape[0] % self.events_channel == 0:
                h, w = evs_data.shape[-2:]
                evs_data = evs_data.view(fold_factor, self.events_channel, h, w).sum(0)
        return evs_data

    def _select_event_paths(self, evs_sample, sample_t, event_mode):
        if event_mode == 'frame':
            return [evs_sample[st] for st in sample_t]
        return evs_sample

    def data_loading(self, paths, sample_t):
        folder_name, rgb_name, rgb_sample, evs_sample, event_mode = paths
        im0 = self.imreader(rgb_sample[0])
        im1 = self.imreader(rgb_sample[-1])
        events = self.ereader(self._select_event_paths(evs_sample, sample_t, event_mode))
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
            im0, im1, events = (
                im0[:, hs:hs + self.crop_size, ws:ws + self.crop_size],
                im1[:, hs:hs + self.crop_size, ws:ws + self.crop_size],
                events[:, hs:hs + self.crop_size, ws:ws + self.crop_size],
            )
            gts = [gt[:, hs:hs + self.crop_size, ws:ws + self.crop_size] for gt in gts]
        else:
            hn, wn = h // 32 * 32, w // 32 * 32
            im0, im1, events = im0[:, :hn, :wn], im1[:, :hn, :wn], events[:, :hn, :wn]
            gts = [gt[:, :hn, :wn] for gt in gts]

        gts = torch.cat(gts, 0)
        left_weight = [1 - float(st) / self.interp_ratio for st in sample_t]
        rgb_name = [os.path.splitext(r)[0] for r in rgb_name]
        data_back = {
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
        return data_back
