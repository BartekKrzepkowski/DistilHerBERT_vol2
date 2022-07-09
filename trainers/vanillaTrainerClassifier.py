import os
import datetime

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from logger.logger import NeptuneLogger
from trainers.utils import get_acc, dynamically_freeze_layers
from trainers.tensorboard_pytorch import TensorboardPyTorch


class VanillaTrainerClassifier(object):
    def __init__(self, model, tokenizer, loaders,
                 optim, accelerator=None, scheduler=None, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.optim = optim
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.loaders = loaders
        self.n_logger = None  # neptune logger
        self.t_logger = None  # tensorflow logger
        self.device = device

    def run_exp(self, epoch_start, epoch_end, exp_name, config_run_epoch, random_seed=42):
        save_path = self.at_exp_start(exp_name, random_seed)
        for epoch in tqdm(range(epoch_start, epoch_end), desc='run_exp'):
            self.model.train()
            self.run_epoch(epoch, save_path, config_run_epoch, phase='train')
            self.model.eval()
            with torch.no_grad():
                self.run_epoch(epoch, save_path, config_run_epoch, phase='test')

    def at_exp_start(self, exp_name, random_seed):
        self.manual_seed(random_seed)
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = os.path.join(os.getcwd(), f'classification/{exp_name}/{date}')
        save_path = f'{base_path}/checkpoints'
        os.makedirs(save_path)
        self.t_logger = TensorboardPyTorch(f'{base_path}/tensorboard', self.device)
        self.n_logger = NeptuneLogger(exp_name)
        return save_path

    def run_epoch(self, epoch, save_path, config_run_epoch, phase):
        running_loss = 0.0
        running_acc = 0.0
        final_acc = 0.0
        running_denom = 0.0
        loader_size = len(self.loaders[phase])
        dynamically_freeze_layers(self.model, epoch, step=2) # from the bottom
        self.n_logger['epoch'].log(epoch)
        progress_bar = tqdm(self.loaders[phase], desc=f'run_epoch: {phase}', mininterval=30, leave=False, total=loader_size)
        for i, data in enumerate(progress_bar):
            self.n_logger['step'].log(i)

            # data = {k: v.to(self.device) for k, v in data.items()}
            output = self.model(input_ids=data['input_ids'], attention_mask=data['attention_mask'],
                                labels=data['labels'])
            loss = output.loss
            acc = get_acc(output.logits, data['labels'])

            self.n_logger[f'{phase}_every_step_loss'].log(loss.item())
            self.n_logger[f'{phase}_every_step_acc'].log(acc)

            loss /= config_run_epoch.grad_accum_steps
            if 'train' in phase:
                # loss.backward()
                self.accelerator.backward(loss) # jedyne użycie acceleratora w trainerze, razem z clip_grad..
                if (i + 1) % config_run_epoch.grad_accum_steps == 0:
                    # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=5.0)
                    self.accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), 5.0)
                    self.optim.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optim.zero_grad()
            loss *= config_run_epoch.grad_accum_steps

            denom = data['input_ids'].size(0)
            running_loss += loss.item() * denom
            running_acc += acc * denom
            final_acc += acc * denom
            running_denom += denom
            # loggers
            if (i + 1) % config_run_epoch.running_step == 0:
                tmp_loss = running_loss / running_denom
                tmp_acc = running_acc / running_denom

                progress_bar.set_postfix({'loss': tmp_loss, 'acc': tmp_acc})

                self.n_logger[f'Loss/{phase}_running'].log(tmp_loss)
                self.n_logger[f'Acc/{phase}_running'].log(tmp_acc)

                self.t_logger.log_scalar(f'Loss/{phase}_running', round(tmp_loss, 4), i + 1 + epoch * loader_size)
                self.t_logger.log_scalar(f'Acc/{phase}_running', round(tmp_acc, 4), i + 1 + epoch * loader_size)

                running_loss = 0.0
                running_acc = 0.0
                running_denom = 0.0

        final_acc /= len(self.loaders[phase].dataset)
        self.n_logger[f'Acc/{phase}_whole'].log(final_acc)
        self.t_logger.log_scalar(f'Acc/{phase}_whole', round(final_acc, 4), epoch)

            # if (i + 1) % config_run_epoch.save_interval == 0:
            #     self.save_student(save_path)

    def save_student(self, path):
        torch.save(self.model.state_dict(), f"{path}/model_{datetime.datetime.utcnow()}.pth")

    def manual_seed(self, random_seed):
        # from transformers import set_seed
        import numpy as np
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        # set_seed(random_seed)
        if 'cuda' in self.device.type:
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
