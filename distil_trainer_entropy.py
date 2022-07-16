import torch
import accelerate
from accelerate import Accelerator
from torch import nn
from models.collator import DataCollatorForWholeWordMask
from datasets import load_from_disk
from torch.utils.data import DataLoader
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EPOCHS = 2
BATCH_SIZE = 10
GRAD_ACCUM_STEPS = 1000 // BATCH_SIZE


def get_dataloaders(tokenizer, path_tokenized_dataset):
    tokenized_datasets = load_from_disk(path_tokenized_dataset)
    train_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=0.15)
    test_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=False)
    train_set = tokenized_datasets['train']
    test_set = tokenized_datasets['test']
    train = DataLoader(dataset=train_set, shuffle=True, batch_size=BATCH_SIZE, collate_fn=train_collator)
    test = DataLoader(dataset=test_set, shuffle=False, batch_size=BATCH_SIZE, collate_fn=test_collator)

    return train, test


def run():
    # init accelerator
    accelerator = Accelerator()
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    device = accelerator.device

    from trainers.utils_model import get_teacher_student_tokenizer
    teacher, student, tokenizer = get_teacher_student_tokenizer()
    train_loader, test_loader = get_dataloaders(tokenizer, 'data/tokenized_dataset_demo2')

    # set accelerator
    from transformers import AdamW, get_cosine_schedule_with_warmup
    from trainers.utils_model import configure_optimizer

    deepspeed_cond = lambda x: deepspeed_plugin is None or x not in deepspeed_plugin.deepspeed_config
    optim_wrapper = AdamW if deepspeed_cond("optimizer") else accelerate.utils.DummyOptim
    optim = configure_optimizer(optim_wrapper, student, None, lr_backbone=5e-5, lr_head=None, weight_decay=1e-3)
    NUM_TRAINING_STEPS = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    print('Num steps', NUM_TRAINING_STEPS, int(0.01 * NUM_TRAINING_STEPS))
    if deepspeed_cond("scheduler"):
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optim,
            num_cycles=EPOCHS,
            num_warmup_steps=int(0.01 * NUM_TRAINING_STEPS),
            num_training_steps=NUM_TRAINING_STEPS)
    else:
        lr_scheduler = accelerate.utils.DummyScheduler(optim, warmup_max_lr=5e-5,
                                                    warmup_num_steps=int(0.01 * NUM_TRAINING_STEPS))

    train_loader, test_loader, teacher, student, optim, lr_scheduler = accelerator.prepare(
        train_loader, test_loader, teacher, student, optim, lr_scheduler)

    loaders = {'train': train_loader, 'test': test_loader}
    print('optim', optim)
    print('lr_scheduler', lr_scheduler)

    from trainers.distilTrainer_hf_collator import DistilTrainer
    params_trainer = {
        'teacher': teacher,#.to(device),
        'student': student,#.to(device),
        'tokenizer': tokenizer,
        'loaders': loaders,
        'criterion1': nn.CrossEntropyLoss().to(device),
        'criterion2': nn.CrossEntropyLoss().to(device),
        # 'criterion2': nn.KLDivLoss('batchmean').to(device), # mam używać log_target?
        'criterion3': nn.CosineEmbeddingLoss().to(device),
        'optim': optim,
        'lr_scheduler': lr_scheduler,
        'accelerator': accelerator,
        'device': device
    }
    trainer = DistilTrainer(**params_trainer)

    import collections
    config_run_epoch = collections.namedtuple('RE', ['save_interval', 'grad_accum_steps', 'running_step'])(20,
                                                                                                           GRAD_ACCUM_STEPS,
                                                                                                           30)

    params_run = {
        'epoch_start': 0,
        'epoch_end': EPOCHS,
        'exp_name': f'ZeRO2, WWM',
        'config_run_epoch': config_run_epoch,
        'temp': 3.0,
        'random_seed': 42
    }

    trainer.run_exp(**params_run)
    trainer.n_logger.run.stop()

if __name__ == '__main__':
    run()
