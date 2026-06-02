import os
import csv
import regex
import jsonlines
from tqdm import tqdm


def jsonlines_load(fname):
    with jsonlines.open(fname, mode='r') as reader:
        data = [row for row in reader]
    return data

def jsonlines_dump(data, fname):
    with jsonlines.open(fname, mode='w') as writer:
        writer.write_all(data)


def _csv_load(fname):
    with open(fname, mode='r', encoding='utf-8') as fin:
        return [row for row in csv.DictReader(fin)]


def _strip_math_boxed(text):
    if text is None:
        return None
    text = str(text)
    marker = '\\boxed'
    pos = text.rfind(marker)
    if pos == -1:
        return text.strip()

    brace = text.find('{', pos + len(marker))
    if brace == -1:
        return text[pos + len(marker):].strip()

    depth = 0
    for idx in range(brace, len(text)):
        char = text[idx]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[brace + 1:idx].strip()
    return text[brace + 1:].strip()


def _normalize_math_answer(answer):
    if answer is None:
        return None
    answer = _strip_math_boxed(answer)
    answer = str(answer).strip()
    for token in ('Final Answer:', 'The final answer is', 'Answer:'):
        if token in answer:
            answer = answer.split(token)[-1].strip()
    answer = answer.strip().strip('$').strip()
    answer = answer.rstrip('.')
    return answer


def normalize_dataset_example(dt_name, example):
    row = dict(example)

    if dt_name == 'math':
        question = row.get('question') or row.get('problem')
        answer = row.get('answer')
        if answer is None and row.get('solution') is not None:
            answer = _normalize_math_answer(row.get('solution'))
        else:
            answer = _normalize_math_answer(answer)
        row.update({
            'question': question,
            'answer': answer,
            'type': row.get('type', 'math'),
        })
        return row

    if dt_name == 'truthfulqa':
        question = row.get('question') or row.get('Question')
        best_answer = row.get('answer') or row.get('Best Answer')
        correct_answers = row.get('Correct Answers') or row.get('correct_answers')
        incorrect_answers = row.get('Incorrect Answers') or row.get('incorrect_answers')
        row.update({
            'question': question,
            'answer': best_answer,
            'correct_answers': correct_answers,
            'incorrect_answers': incorrect_answers,
            'category': row.get('Category') or row.get('category'),
        })
        return row

    if dt_name in ['gsm8k', 'gsm8k_cot'] and isinstance(row.get('answer'), str) and '####' in row['answer']:
        row['answer'] = row['answer'].split('####')[-1].strip().replace(',', '')

    if dt_name == 'strategyqa' and isinstance(row.get('answer'), bool):
        row['answer'] = 'yes' if row['answer'] else 'no'

    return row


def load_dataset_examples(dt_name, input_file):
    if dt_name == 'truthfulqa' and input_file.lower().endswith('.csv'):
        data = _csv_load(input_file)
    else:
        data = jsonlines_load(input_file)
    return [normalize_dataset_example(dt_name, row) for row in data]


def _merge_sub(fn_prefix: str):
    main_data = jsonlines_load(f'{fn_prefix}.jsonl')
    sub_data = jsonlines_load(f'{fn_prefix}_sub.jsonl')
    
    indexes = [d['index'] for d in main_data if 'index' in d]
    for d in sub_data:
        if 'index' not in d: continue
        if d['index'] in indexes: continue
        main_data.append(d)

    main_data = main_data[:1] + sorted(main_data[1:], key=lambda x: x['index'])
    jsonlines_dump(main_data, f'{fn_prefix}.jsonl')

def merge_parallel_results(fn_prefix: str, num: int):
    results, indexes = [], []
    if os.path.exists(f'{fn_prefix}.jsonl'):
        results = jsonlines_load(f'{fn_prefix}.jsonl')
        indexes = [d['index'] for d in results if 'index' in d]
    for i in tqdm(range(num)):
        fname = f'{fn_prefix}_parallel_split{i}.jsonl'
        if os.path.exists(fname):
            results += [x for x in jsonlines_load(fname) if x not in results and ('index' not in x or x['index'] not in indexes)]
    if 'index' in results[-1]:
        if 'index' not in results[0]:
            results = [results[0]] + sorted(results[1:], key=lambda x: x['index'])
        else:
            results.sort(key=lambda x: x['index'])
    with jsonlines.open(f'{fn_prefix}.jsonl', mode='w') as writer:
        writer.write_all(results)
    
    if fn_prefix.endswith('_sub'):
        _merge_sub(fn_prefix[:-4])


def split_parallel_results(fn_prefix: str, num: int):
    results = jsonlines_load(f'{fn_prefix}.jsonl')
    prompt = results[0]
    sid = int(regex.search(r'_s\d+_', fn_prefix).group().strip('s_'))
    eid = int(regex.search(r'_e\d+_', fn_prefix).group().strip('e_'))
    step = (eid - sid + num - 1) // num
    to_dump = [[] for _ in range(num)]
    to_dump[0].append(prompt)
    for rst in results[1:]:
        idx = rst['index']
        fid = idx // step
        to_dump[fid].append(rst)
    for fid, data in enumerate(to_dump):
        if not len(data): continue
        with jsonlines.open(f'{fn_prefix}_parallel_split{fid}.jsonl', mode='w') as writer:
            writer.write_all(data)
            
