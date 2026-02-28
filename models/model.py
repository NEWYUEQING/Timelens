import os
import time
from copy import deepcopy

import losses
import numpy as np
import torch
from torch.nn import Module
from torchvision.transforms import ToPILImage

from tools.registery import LOSS_REGISTRY

mkdir = lambda x: os.makedirs(x, exist_ok=True)


class BaseModel(Module):
    """Minimal training/validation base model without flow-specific branches."""

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.train_im_path = params.paths.save.train_im_path
        self.val_im_path = os.path.join(params.paths.save.val_im_path, "images")
        mkdir(self.val_im_path)

        self.record_txt = params.paths.save.record_txt
        self.val_record_txt = os.path.join(self.val_im_path, 'detailed_records')
        self.training_metrics = {}
        self.validation_metrics = {}

        self.train_print_freq = params.training_config.train_stats.print_freq
        self.train_im_save = params.training_config.train_stats.save_im_ep
        self.val_eval = params.validation_config.weights_save_freq
        self.val_im_save = params.validation_config.val_imsave_epochs
        self.interp_num = params.training_config.interp_ratio - 1

        self.toim = ToPILImage()
        self.metrics_init()
        self.params_training = None
        self.debug = params.debug
        self.save_images = params.save_images

    def write_log(self, logcont):
        with open(self.record_txt, 'a+') as f:
            f.write(logcont)

    def _init_metrics(self, metrics):
        if metrics is None:
            metrics = {}
            for k in self.params.training_config.losses.keys():
                metrics.update({f"train_{k}": []})
            for k in self.params.validation_config.losses.keys():
                metrics.update({f"val_{k}": []})
        self.metrics = metrics
        self.metrics_record = deepcopy(metrics)
        self.metrics_record['training_time'] = []
        self.metrics_record['validation_time'] = []

    def metrics_init(self):
        for k in self.params.training_config.losses.keys():
            self.training_metrics.update({
                k: [LOSS_REGISTRY.get(k)(self.params.training_config.losses[k]),
                    self.params.training_config.losses[k]['as_loss']]
            })
        for k in self.params.validation_config.losses.keys():
            self.validation_metrics.update({
                k: LOSS_REGISTRY.get(k)(self.params.validation_config.losses[k])
            })

    def _update_training_time(self):
        self.metrics_record['training_time'].append(time.time())

    def _update_validation_time(self):
        self.metrics_record['validation_time'].append(time.time())

    def _print_train_log(self, epoch):
        print_content = f"EPOCH/MAX EPOCH: {epoch}/{self.params.training_config.max_epoch}\t"
        for k in self.training_metrics.keys():
            print_content += f'{k}:{np.mean(self.metrics_record[f"train_{k}"]):.4f}\t'
        print_content = print_content.strip('\t') + '\n'
        print(print_content)
        return print_content

    def _print_val_log(self):
        print_content = "Validation Logs: \t"
        for k in self.validation_metrics.keys():
            print_content += f'{k}:{np.mean(self.metrics_record[f"val_{k}"]):.4f}\t'
        print_content = print_content.strip('\t') + '\n'
        print(print_content)
        return print_content

    def net_training(self, data_in, optim, epoch, step):
        pass

    def validation(self, data_in, epoch):
        pass

    def forward(self, *args, **kwargs):
        pass

    def update_training_metrics(self, res, gt, epoch, step, lr, *args, **kwargs):
        loss = 0
        print_content = f"MODEL {self.params.model_config.name}\tCur EPOCH/STEP/LR: [{epoch}/{step}/{lr:.6f}]\t"
        self.metrics_record["training_time"].append(time.time())

        for k in self.training_metrics.keys():
            func, as_loss = self.training_metrics[k]
            if k == 'ECharbonier':
                events = args[0]
                loss_item = func.forward(res, gt, events)
            elif k == "AnnealCharbonier":
                loss_item = func.forward(res, gt, kwargs['fuseout'], epoch)
            elif k in ('dists', 'lpips') and len(res.shape) == 5:
                _, _, rc, rh, rw = res.shape
                loss_item = func.forward(res.view(-1, rc, rh, rw), gt.view(-1, rc, rh, rw))
            else:
                loss_item = func.forward(res, gt)

            if as_loss:
                loss += loss_item
            self.metrics_record[f"train_{k}"].append(loss_item.item())

            if step % self.train_print_freq == 0:
                print_content += f'{k}: {self.metrics_record[f"train_{k}"][-1]:.4f}\t'

        if step % self.train_print_freq == 0:
            print(print_content)
            with open(os.path.join(self.train_im_path, str(epoch) + '.txt'), 'a+') as f:
                f.write(print_content + '\n')
        return loss

    def update_validation_metrics(self, res, gt, epoch, data_in, *args, **kwargs):
        os.makedirs(self.val_record_txt, exist_ok=True)
        for n in range(res.shape[1]):
            detailed_record = f'EPOCH {epoch}\tFolder: {data_in["folder"][0]} Image: {data_in["rgb_name"][n]} val num: {n}\t'
            for k in self.validation_metrics.keys():
                val = self.validation_metrics[k].forward(res[:, n].detach(), gt[:, n].detach()).item()
                self.metrics_record[f'val_{k}'].append(val)
                detailed_record += f'{k}: {val:.4f}\t'

            with open(os.path.join(self.val_record_txt, f"{epoch}.txt"), 'a+') as f:
                f.write(detailed_record.strip('\t') + '\n')

            if (epoch % max(self.params.validation_config.val_imsave_epochs, 1) == 0 and self.save_images) or not self.params.enable_training:
                os.makedirs(os.path.join(self.val_im_path, str(epoch)), exist_ok=True)
                rgb_name = data_in['rgb_name']
                folder = os.path.split(data_in['folder'][0])[-1]
                self.toim(res[0, n].detach().cpu().clamp(0, 1)).save(
                    os.path.join(self.val_im_path, str(epoch), f"{folder}_{rgb_name[n+1][0]}_{n}_res.jpg")
                )
                self.toim(gt[0, n]).save(
                    os.path.join(self.val_im_path, str(epoch), f"{folder}_{rgb_name[n+1][0]}_{n}_gt.jpg")
                )


class OursBase(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.grad_cache = {}
        self.real_interp = None if 'real_interp' not in params.keys() else params.real_interp

    def net_training(self, data_in, optim, epoch, step):
        self.train()
        optim.zero_grad()
        left_frame, right_frame, events = data_in['im0'].cuda(), data_in['im1'].cuda(), data_in['events'].cuda()
        interp_ratio = data_in['interp_ratio'][0].item()

        gts = data_in['gts'].cuda().unsqueeze(2)
        scalar = interp_ratio - 1 if self.real_interp is None else self.real_interp - 1
        n, _, _, h, w = gts.shape
        gts = gts.reshape(n, scalar, -1, h, w)

        res = self.forward(left_frame, right_frame, events, interp_ratio)
        recon = res[0] if isinstance(res, (list, tuple)) else res

        fuseout = res[-1] if isinstance(res, (list, tuple)) else recon
        loss = self.update_training_metrics(recon, gts, epoch, step, optim.param_groups[0]['lr'], events, fuseout=fuseout)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.01)
        optim.step()

        if epoch % self.train_im_save == 0 and step % self.train_print_freq == 0:
            self.save_training_samples(recon, gts, data_in, epoch, step)

    def save_training_samples(self, res, gt, data_in, epoch, step):
        save_folder = os.path.join(self.train_im_path, str(epoch), str(step))
        mkdir(save_folder)
        file_names = data_in['rgb_name']
        n_total, b_total = len(file_names), len(file_names[0])

        for n in range(n_total):
            for b in range(b_total):
                if n == 0:
                    self.toim(data_in['im0'][b]).save(os.path.join(save_folder, f"b{b}n{n}_im0_id{file_names[n][b]}.jpg"))
                elif n == n_total - 1:
                    self.toim(data_in['im1'][b]).save(os.path.join(save_folder, f"b{b}n{n}_im1_id{file_names[n][b]}.jpg"))
                else:
                    self.toim((res[b, n - 1]).clamp(0, 1)).save(os.path.join(save_folder, f"b{b}n{n}_res_id{file_names[n][b]}.jpg"))
                    self.toim(gt[b, n - 1]).save(os.path.join(save_folder, f"b{b}n{n}_gt_id{file_names[n][b]}.jpg"))

    def net_validation(self, data_in, epoch):
        self.eval()
        with torch.no_grad():
            left_frame, right_frame, events = data_in['im0'].cuda(), data_in['im1'].cuda(), data_in['events'].cuda()
            interp_ratio = data_in['interp_ratio'].item()

            gts = data_in['gts'].cuda().unsqueeze(2)
            scalar = interp_ratio - 1 if self.real_interp is None else self.real_interp - 1
            n, _, _, h, w = gts.shape
            gts = gts.reshape(n, scalar, -1, h, w)

            res = self.forward(left_frame, right_frame, events, interp_ratio)
            recon = res[0] if isinstance(res, (list, tuple)) else res
            self.update_validation_metrics(recon, gts, epoch, data_in)

    def forward(self, left_frame, right_frame, events, interp_ratio):
        real_interp = interp_ratio if self.real_interp is None else self.real_interp
        jump_ratio = interp_ratio // real_interp
        end_tlist = range(jump_ratio - 1, interp_ratio - 1, jump_ratio)
        return self.net(torch.cat((left_frame, right_frame), 1), events, interp_ratio, end_tlist)
