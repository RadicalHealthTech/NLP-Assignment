import pdb
import numpy as np
import random
import math
import config
import json
import time
import os
from tokenizer import tokenize
import torch
import torch.nn as nn
import torch.optim as optim

from itertools import chain
from torchtext.data import BucketIterator, Iterator, Field, Dataset, Example
from torchtext.vocab import GloVe


def read_squad_json(path, sample):
    """ Reads the squad train/test json format into individual context-question-answer pairs"""
    with open(path) as f:
        data = json.load(f)
        data = data["data"]

    cqa_list = []
    for i, title in enumerate(data):
        if sample and i == 2:
            break
        for para in title["paragraphs"]:
            context = para["context"]
            qas = para["qas"]
            for qa in qas:
                is_impossible = qa["is_impossible"]
                question = qa["question"]
                qa_id = qa["id"]

                if not is_impossible:
                    answer_text = qa["answers"][0]["text"]
                    answer_start = qa["answers"][0]["answer_start"]
                else:
                    answer_text = " "
                    answer_start = -1

                context = [x.txt.lower() for x in tokenize(context) if x.txt]
                question = [x.txt.lower() for x in tokenize(question) if x.txt]
                answer_text = [x.txt.lower() for x in tokenize(answer_text) if x.txt]

                current_cqa = {
                    "context": context,
                    "question": question,
                    "is_impossible": is_impossible,
                    "answer_text": answer_text,
                    "answer_start": answer_start,
                    "qa_id": qa_id,
                }

                cqa_list.append(current_cqa)
    return cqa_list


def context_plus_question_seq(context, question):
    """ Combine the Context and Question using the question start seperator to get the input format"""
    return list(context) + ["<q>"] + list(question)  # TODO remove ?


def answer_seq(answer):
    """ Convert answer generator to get the answer format"""
    return list(answer)


def convert_data_to_input_format(data):

    examples = []
    for cqa in data:
        context = cqa["context"]
        question = cqa["question"]
        answer = cqa["answer_text"]

        source = context_plus_question_seq(context, question)
        answer = answer_seq(answer)

        # examples.append({"source": source, "answer": answer})
        examples.append((source, answer))
    return examples


class SquadDataset(Dataset):
    """ Squad dataset for train/test
        Input: Json file path, torch.text.Field object
        Output: torch.data.Dataset object
    """

    def __init__(self, json_file, text_field):
        fields = [("source", text_field), ("answer", text_field)]
        self.sort_key = lambda x: len(x.source)

        data = read_squad_json(json_file, sample=config.sample)
        examples = convert_data_to_input_format(data)
        examples = [Example.fromlist(example, fields) for example in examples]

        super(SquadDataset, self).__init__(examples, fields)
