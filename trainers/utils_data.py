import glob
import multiprocessing
from datasets import concatenate_datasets, load_from_disk, load_dataset

PATH_IN = 'data/cc100_demo.txt'
PATH_OUT = 'data/cc100_filtered_demo'

def preprocess_dataset(path_in, path_out):
    raw_datasets = load_dataset('text', data_files=path_in)
    NUM_PROC = multiprocessing.cpu_count()

    import re
    import html as ihtml
    from bs4 import BeautifulSoup

    def clean_text(text):
        text = BeautifulSoup(ihtml.unescape(text), "lxml").text
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    filter_non_alfanum = lambda x: re.sub('[^0-9AaĄąBbCcĆćDdEeĘęFfGgHhIiJjKkLlŁłMmNnŃńOoÓóPpRrSsŚśTtUuWwYyZzŹźŻż\,\. ]+', '', x)
    filter_ratio = lambda x: len(filter_non_alfanum(x)) / len(x)

    raw_datasets = raw_datasets.filter(lambda x: len(x['text']) > 15, num_proc=NUM_PROC)
    raw_datasets = raw_datasets.map(lambda x: {'text':  [clean_text(y) for y in x['text']]}, batched=True, num_proc=NUM_PROC)
    raw_datasets = raw_datasets.filter(lambda x: len(x['text']) > 15 and filter_ratio(x['text']) > 0.9, num_proc=NUM_PROC)
    raw_datasets.save_to_disk(path_out)

preprocess_dataset(PATH_IN, PATH_OUT)


# ver1
def tokenize_dataset1(path_dedup_dataset, path_tokenized_out, is_dedup):
    from transformers.models.herbert.tokenization_herbert_fast import HerbertTokenizerFast
    tokenizer = HerbertTokenizerFast.from_pretrained("allegro/herbert-base-cased")
    NUM_PROC = multiprocessing.cpu_count()
    if is_dedup:
        dedup_datasets = [load_dataset('json', data_files=path)['train'] for path in glob.glob(path_dedup_dataset)] #'./datasets/data/*.json.gz'
        dedup_dataset = concatenate_datasets(dedup_datasets)
        dedup_dataset = dedup_dataset.remove_columns(['text', 'token_type_ids'])
    else:
        dedup_dataset = load_from_disk(path_dedup_dataset)
    def tokenize_function(example):
        tokenized = tokenizer(example['text'], truncation=True)
        return tokenized

    tokenized_dataset = dedup_dataset.map(tokenize_function, batched=True, num_proc=NUM_PROC)
    tokenized_dataset = tokenized_dataset.remove_columns(['text', 'token_type_ids'])
    tokenized_dataset = tokenized_dataset.with_format('torch')
    tokenized_dataset = tokenized_dataset['train'].train_test_split(test_size=0.01, seed=29)
    print(tokenized_dataset)
    tokenized_dataset.save_to_disk(path_tokenized_out)

# tokenize_dataset1(dedup_datasets, 'data/tokenized_dataset_demo')



# ver2
def get_proper_idx1(idx, context_length, words_ids):
    if idx + context_length >= len(words_ids) - 1:
        return idx + context_length, idx + context_length
    if words_ids[idx + context_length] != words_ids[idx + context_length - 1]:
        return idx + context_length, idx + context_length
    else:
        while words_ids[idx + context_length] == words_ids[idx + context_length - 1]:
            idx -= 1
        return idx + context_length, idx + context_length

def get_proper_idx2(idx, context_length, words_ids):
    if idx + context_length >= len(words_ids) - 1:
        return idx + context_length, idx + context_length
    if words_ids[idx + context_length - 1] == None:
        return idx + context_length, idx + context_length
    else:
        while words_ids[idx + context_length] == words_ids[idx + context_length - 1]:
            idx -= 1
        lidx = idx
        ridx = idx
        while words_ids[lidx + context_length - 1] != None:
            lidx -= 1
        while words_ids[ridx + context_length - 1] != None:
            ridx += 1
        lidx = lidx + context_length
        ridx = ridx + context_length
        idx = idx + context_length

        if idx - lidx < 20:
            return lidx, lidx
        elif ridx - idx < 20:
            return idx, ridx
        else:
            return idx, idx

def tokenize_dataset2(path_dedup_dataset, path_tokenized_out, is_dedup):
    from transformers.models.herbert.tokenization_herbert_fast import HerbertTokenizerFast
    tokenizer = HerbertTokenizerFast.from_pretrained("allegro/herbert-base-cased")
    context_length = tokenizer.model_max_length
    if is_dedup:
        dedup_datasets = [load_dataset('json', data_files=path)['train'] for path in glob.glob(path_dedup_dataset)] #'./datasets/data/*.json.gz'
        dedup_dataset = concatenate_datasets(dedup_datasets)
        dedup_dataset = dedup_dataset.remove_columns(['text', 'token_type_ids'])
    else:
        dedup_dataset = load_from_disk(path_dedup_dataset)

    NUM_PROC = multiprocessing.cpu_count()
    def tokenize_function(example):
        all_input_ids = [0]
        all_words_ids = [None]
        results = tokenizer(example['text'], add_special_tokens=False)
        for i, input_ids in enumerate(results['input_ids']):
            all_input_ids.extend(input_ids)
            all_input_ids.append(tokenizer.sep_token_id)

            all_words_ids.extend(results.word_ids(i))
            all_words_ids.append(None)
        chunks1 = []
        chunks2 = []
        i = 0
        while i < len(all_input_ids):
            j_min, j_max = get_proper_idx2(i, context_length, all_words_ids)
            chunks1.append([0] + all_input_ids[i: j_min])
            chunks2.append([None] + all_words_ids[i: j_min])
            i = j_max
        return {'input_ids': chunks1, 'word_ids': chunks2}

    tokenized_dataset = dedup_dataset.map(tokenize_function, batched=True, num_proc=NUM_PROC, remove_columns=['text'])
    # tokenized_dataset = tokenized_dataset.remove_columns(['text', 'token_type_ids'])
    # tokenized_dataset = tokenized_dataset.with_format('torch')
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) >= 30, num_proc=NUM_PROC)
    tokenized_dataset = tokenized_dataset['train'].train_test_split(test_size=0.01, seed=29)
    print(tokenized_dataset)
    tokenized_dataset.save_to_disk(path_tokenized_out)

tokenize_dataset2('./datasets/data/*.json.gz', 'data/tokenized_dataset_demo2', True)
