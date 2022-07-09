import torch
from accelerate import Accelerator
from torch import nn
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from datasets import load_from_disk
from torch.utils.data import DataLoader
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 2
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 1024 // BATCH_SIZE


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
    accelerator = Accelerator(device_placement=True, fp16=True, mixed_precision='fp16')
    device = accelerator.device

    from trainers.utils import get_teacher_student_tokenizer, print_model_size
    teacher, student, tokenizer = get_teacher_student_tokenizer()
    train_loader, test_loader = get_dataloaders(tokenizer, 'datasets/tokenized_dataset')

    # set accelerator
    from transformers import AdamW, get_cosine_schedule_with_warmup
    from trainers.utils import configure_optimizer

    optim = configure_optimizer(AdamW, student, None, lr_backbone=5e-5, lr_head=None, weight_decay=1e-3)

    train_loader, test_loader, teacher, student, optim = accelerator.prepare(
        train_loader, test_loader, teacher, student, optim)

    loaders = {'train': train_loader, 'test': test_loader}

    NUM_TRAINING_STEPS = len(train_loader) // GRAD_ACCUM_STEPS * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optim,
        num_cycles=EPOCHS,
        num_warmup_steps=int(0.01 * NUM_TRAINING_STEPS),
        num_training_steps=NUM_TRAINING_STEPS)

    from trainers.distilTrainer_hf_collator import DistilTrainer

    params_trainer = {
        'teacher': teacher.to(device),
        'student': student.to(device),
        'tokenizer': tokenizer,
        'loaders': loaders,
        'criterion1': nn.CrossEntropyLoss().to(device),
        'criterion2': nn.CrossEntropyLoss().to(device),
        # 'criterion2': nn.KLDivLoss('batchmean').to(device), # mam używać log_target?
        'criterion3': nn.CosineEmbeddingLoss().to(device),
        'optim': optim,
        'scheduler': scheduler,
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
        'exp_name': f'plain_distil_hf_collator,preprocessing,deduplication,whole_word_masking_hf',
        'config_run_epoch': config_run_epoch,
        'temp': 2.0,
        'random_seed': 42
    }

    trainer.run_exp(**params_run)
    trainer.n_logger.run.stop()

if __name__ == '__main__':
    run()