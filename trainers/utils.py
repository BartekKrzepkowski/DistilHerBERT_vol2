import os

import torch
import numpy as np

from models.distil_student import creat_student


def dynamically_freeze_layers(model, epoch, step):
    epoch -= 2
    if epoch >= 0:
        for name, params in model.bert.embeddings.named_parameters():
            if params.requires_grad:
                params.requires_grad = False
        for name, params in model.bert.encoder.layer.named_parameters():
            layers_range = range(0, epoch // step * step)
            if params.requires_grad and any(str(c) == name.split('.')[0] for c in layers_range):
                params.requires_grad = False


def dynamically_unfreeze_layers(model, epoch, step):
    epoch = 12 - epoch + 1
    if epoch >= 0:
        for name, params in model.bert.encoder.layer.named_parameters():
            layers_range = range(epoch // step * step, 12)
            if not params.requires_grad and any(str(c) == name.split('.')[0] for c in layers_range):
                params.requires_grad = True
    if epoch == -2:
        for name, params in model.bert.embeddings.named_parameters():
            if not params.requires_grad:
                params.requires_grad = True
    # for name, params in model.named_parameters():
    #     if params.requires_grad:
    #         print(name, params.requires_grad)


def get_acc(y_pred, y_true):
    acc = (y_pred.argmax(dim=1) == y_true).float().mean().item()
    return acc

# czy powinniśmy zasłaniać unk token?
def create_masked_ids(data):
    mask1 = torch.rand(data.input_ids.shape) < 0.15
    mask2 = torch.tensor(~np.isin(data.input_ids.detach().cpu().numpy(), (0, 1, 2)))
    masked_ids = (mask1 * mask2)
    data.input_ids[masked_ids] = 4
    return masked_ids


def get_masked_mask_and_att_mask(labels, lengths):
    mask1 = torch.tensor(~np.isin(labels.detach().cpu().numpy(), (0, 1, 2, 4)))
    att_mask = torch.arange(mask1.size(1))[None, :] < lengths[:, None].detach().cpu()
    return mask1, att_mask


def get_masked_mask_hf_collator(labels):
    mask1 = labels > 0
    return mask1


def get_teacher_student_tokenizer():
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    teacher = AutoModel.from_pretrained("allegro/herbert-base-cased")
    student = creat_student()
    
    print(f'Number of parameters: Teacher: {count_parameters(teacher)}, Student: {count_parameters(student)},'
          f'Student / Teacher ratio: {round(count_parameters(student) / count_parameters(teacher), 4)}.')

    for params in teacher.parameters():
        params.requires_grad = False

    return teacher, student, tokenizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

# backbone, head, add dicts, 10% lr
def configure_optimizer(optim, backbone, head, lr_backbone, lr_head, weight_decay=1e-4, **optim_kwargs):
    alert_chunks = ['embeddings', 'LayerNorm', 'bias']
    no_decay = {pn for pn, p in backbone.named_parameters() if any(c in pn for c in alert_chunks)}
    optimizer_grouped_parameters = [
        {
            "params": [p for pn, p in backbone.named_parameters() if pn not in no_decay and p.requires_grad],
            "weight_decay": weight_decay,
            'lr': lr_backbone
        },
        {
            "params": [p for pn, p in backbone.named_parameters() if pn in no_decay and p.requires_grad],
            "weight_decay": 0.0,
            'lr': lr_backbone
        },
    ]
    if head is not None:
        no_decay = {pn for pn, p in head.named_parameters() if any(c in pn for c in alert_chunks)}
        optimizer_grouped_parameters2 = [
            {
                "params": [p for pn, p in head.named_parameters() if pn not in no_decay and p.requires_grad],
                "weight_decay": weight_decay,
                'lr': lr_head
            },
            {
                "params": [p for pn, p in head.named_parameters() if pn in no_decay and p.requires_grad],
                "weight_decay": 0.0,
                'lr': lr_head
            },
        ]
        optimizer_grouped_parameters += optimizer_grouped_parameters2

    optimizer = optim(optimizer_grouped_parameters, **optim_kwargs)
    return optimizer
