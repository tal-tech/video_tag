#coding=utf-8
import os
import sys
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path))
import json
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from transformers import AlbertModel, BertTokenizer, AlbertConfig

__all__ = ["interactive_detector"]


class BeikeModel(nn.Module):

    def __init__(self, pretrained_path, num_classes, conf):
        super(BeikeModel, self).__init__()

        self.pretrained_model = AlbertModel.from_pretrained(pretrained_path, config=conf)

        self.linear1 = nn.Linear(conf.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids, target_mask):
        '''
        Args:
            target_mask: 标识出每个输入中，目标文本所在的时间步。属于目标文本的为1，不属于的为0。[batch_size, seq_len]
        '''

        _, _, hidden_states = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        out = hidden_states[-1]

        target_mask = target_mask.unsqueeze(dim=-1)  # [batch_size, seq_len, 1]
        target_out = out * target_mask  # [batch_size, seq_len, hidden_size]
        target_out = target_out.sum(dim=1) / target_mask.sum(dim=1)  # [batch_size, hidden_size] / [batch_size, 1]

        logits = self.linear1(target_out)  # [batch_size, num_classes]

        return logits


class TextDataset(Dataset):

    def __init__(self, examples, attention_masks, token_type_ids, target_sentence_masks,
                 texts, sentence_ids, begin_times, end_times):
        self.examples = examples
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.target_sentence_masks = target_sentence_masks

        self.texts = texts
        self.sentence_ids = sentence_ids
        self.begin_times = begin_times
        self.end_times = end_times

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return {
            "ids": torch.tensor(self.examples[item], dtype=torch.long),
            "attention_masks": torch.tensor(self.attention_masks[item], dtype=torch.float),
            "token_type_ids": torch.tensor(self.token_type_ids[item], dtype=torch.long),
            "target_sentence_masks": torch.tensor(self.target_sentence_masks[item], dtype=torch.float),

            "texts": self.texts[item],
            "sentence_ids": self.sentence_ids[item],
            "begin_times": self.begin_times[item],
            "end_times": self.end_times[item],
        }


class Interactive:

    def __init__(self, model_path, device="cpu"):

        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained(model_path)

        model_config = AlbertConfig.from_pretrained(model_path)
        model_config.output_hidden_states = True

        self.model = BeikeModel(pretrained_path=model_path,
                           num_classes=3,
                           conf=model_config)

        if device == "cuda":
            self.model.load_state_dict(torch.load(os.path.join(model_path, "model.pt")))
        elif device == "cpu":
            self.model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), map_location=torch.device('cpu')))
        else:
            print("ERROR: wrong device")
            raise ValueError("Wrong device argument.")


        self.model.to(device)

        print("Model loading successful.")


    def process_one_example(self, target_text, previous_text=None, next_text=None, block_size=512):
        block_size = block_size - 3

        if previous_text is not None and next_text is not None:
            print("ERROR: both previous and next sentences are received")
            return None

        if previous_text is None and next_text is None:
            print("ERROR: neither previous nor next sentences are received")
            return None

        target_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(target_text))

        if previous_text is not None:
            previous_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(previous_text))
            if len(previous_ids + target_ids) > block_size:
                diff = len(previous_ids + target_ids) - block_size
                previous_ids = previous_ids[diff:]
                if len(previous_ids + target_ids) != block_size:
                    print("WARNING: previous sent and target sent are too long, text length is not correct after truncating. Final length is: {}.".format(len(previous_ids + target_ids)))

            example = self.tokenizer.build_inputs_with_special_tokens(previous_ids, target_ids)
            attention_mask = [1] * len(example)

            type_idx = self.tokenizer.create_token_type_ids_from_sequences(previous_ids, target_ids)

            target_mask = [0] * (len(type_idx) - sum(type_idx)) + [1] * (sum(type_idx) - 1) + [0]

            text = target_text

            return {
                "example": example,
                "attention_mask": attention_mask,
                "type_idx": type_idx,
                "target_mask": target_mask,
                "text": text,
            }

        else:
            next_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(next_text))
            if len(target_ids + next_ids) > block_size:
                diff = len(target_ids + next_ids) - block_size
                next_ids = next_ids[:-diff]
                if len(target_ids + next_ids) != block_size:
                    print("WARNING: target sent and next sent are too long, text length is not correct after truncating. Final length is: {}.".format(len(target_ids + next_ids)))

            example = self.tokenizer.build_inputs_with_special_tokens(target_ids, next_ids)
            attention_mask = [1] * len(example)

            type_idx = self.tokenizer.create_token_type_ids_from_sequences(target_ids, next_ids)

            target_mask = [0] + [1] * (len(type_idx) - sum(type_idx) - 2) + [0] + [0] * (sum(type_idx))

            text = target_text

            return {
                "example": example,
                "attention_mask": attention_mask,
                "type_idx": type_idx,
                "target_mask": target_mask,
                "text": text,
            }


    def process_asr(self, lines):
        examples = []
        attention_masks = []
        token_type_ids = []
        target_sentence_masks = []

        texts = []
        target_sent_ids = []
        target_begin_times = []
        target_end_times = []

        for i in range(len(lines)):
            line = lines[i]
            target_begin = int(line['begin_time'])
            target_end = int(line['end_time'])
            target_sent_id = int(line['sentence_id'])
            target_text = line["text"]

            if target_text.strip() == "":
                continue

            if i == 0 and i == len(lines) - 1:

                print("WARNING: There is only one sentence in input asr.")

                data = self.process_one_example(target_text, next_text="")

                if data is None:
                    continue

                examples.append(data["example"])
                attention_masks.append(data["attention_mask"])
                token_type_ids.append(data["type_idx"])
                target_sentence_masks.append(data["target_mask"])
                texts.append(data["text"])
                target_sent_ids.append(target_sent_id)
                target_begin_times.append(target_begin)
                target_end_times.append(target_end)

                data = self.process_one_example(target_text, previous_text="")

                if data is None:
                    continue

                examples.append(data["example"])
                attention_masks.append(data["attention_mask"])
                token_type_ids.append(data["type_idx"])
                target_sentence_masks.append(data["target_mask"])
                texts.append(data["text"])
                target_sent_ids.append(target_sent_id)
                target_begin_times.append(target_begin)
                target_end_times.append(target_end)

            if i != len(lines) - 1:
                next_line = lines[i + 1]
                next_text = next_line["text"]

                data = self.process_one_example(target_text, next_text=next_text)

                if data is None:
                    continue

                examples.append(data["example"])
                attention_masks.append(data["attention_mask"])
                token_type_ids.append(data["type_idx"])
                target_sentence_masks.append(data["target_mask"])
                texts.append(data["text"])
                target_sent_ids.append(target_sent_id)
                target_begin_times.append(target_begin)
                target_end_times.append(target_end)

            if i != 0:
                previous_line = lines[i - 1]
                previous_text = previous_line["text"]

                data = self.process_one_example(target_text, previous_text=previous_text)

                if data is None:
                    continue

                examples.append(data["example"])
                attention_masks.append(data["attention_mask"])
                token_type_ids.append(data["type_idx"])
                target_sentence_masks.append(data["target_mask"])
                texts.append(data["text"])
                target_sent_ids.append(target_sent_id)
                target_begin_times.append(target_begin)
                target_end_times.append(target_end)


        return TextDataset(examples=examples, attention_masks=attention_masks, token_type_ids=token_type_ids,
                           target_sentence_masks=target_sentence_masks,
                           texts=texts, sentence_ids=target_sent_ids, begin_times=target_begin_times, end_times=target_end_times)

    def inference(self, dataset, inference_batch_size=16):

        def collate(examples):
            '''
            examples: List[Dict]
            '''
            ids = [x["ids"] for x in examples]
            attention_masks = [x["attention_masks"] for x in examples]
            token_type_ids = [x["token_type_ids"] for x in examples]
            target_sentence_masks = [x["target_sentence_masks"] for x in examples]

            texts = [x["texts"] for x in examples]
            sentence_ids = [x["sentence_ids"] for x in examples]
            begin_times = [x["begin_times"] for x in examples]
            end_times = [x["end_times"] for x in examples]

            if self.tokenizer._pad_token is None:
                ids = pad_sequence(ids, batch_first=True)
                attention_masks = pad_sequence(attention_masks, batch_first=True)
                token_type_ids = pad_sequence(token_type_ids, batch_first=True)
                target_sentence_masks = pad_sequence(target_sentence_masks, batch_first=True)

                return {
                    "ids": ids,
                    "attention_masks": attention_masks,
                    "token_type_ids": token_type_ids,
                    "target_sentence_masks": target_sentence_masks,

                    "texts": texts,
                    "sentence_ids": sentence_ids,
                    "begin_times": begin_times,
                    "end_times": end_times,
                }
            else:
                ids = pad_sequence(ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                target_sentence_masks = pad_sequence(target_sentence_masks, batch_first=True,
                                                     padding_value=self.tokenizer.pad_token_id)

                return {
                    "ids": ids,
                    "attention_masks": attention_masks,
                    "token_type_ids": token_type_ids,
                    "target_sentence_masks": target_sentence_masks,

                    "texts": texts,
                    "sentence_ids": sentence_ids,
                    "begin_times": begin_times,
                    "end_times": end_times,
                }

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset, sampler=sampler, batch_size=inference_batch_size, collate_fn=collate
        )
        self.model.eval()

        y_logits = []

        all_texts = []
        all_sentence_ids = []
        all_begin_times = []
        all_end_times = []

        for batch in dataloader:
            inputs = batch["ids"]
            masks = batch["attention_masks"]
            type_ids = batch["token_type_ids"]
            target_masks = batch["target_sentence_masks"]

            texts = batch["texts"]
            sentence_ids = batch["sentence_ids"]
            begin_times = batch["begin_times"]
            end_times = batch["end_times"]

            all_texts += texts
            all_sentence_ids += sentence_ids
            all_begin_times += begin_times
            all_end_times += end_times

            inputs = inputs.to(self.device)
            masks = masks.to(self.device)
            type_ids = type_ids.to(self.device)
            target_masks = target_masks.to(self.device)

            with torch.no_grad():
                logits = self.model(
                    input_ids=inputs,
                    attention_mask=masks,
                    token_type_ids=type_ids,
                    target_mask=target_masks,
                )

                y_logits.append(logits.cpu())

        y_logits = torch.cat(y_logits, dim=0)  # [batch, num_classes]

        try:
            assert len(y_logits) == len(all_texts) == len(all_sentence_ids) == len(all_begin_times) == len(all_end_times)
        except Exception as e:
            print("WARNING: results' length are not equal.")

        sentence_id2result = {}

        for i in range(len(all_sentence_ids)):
            idx = all_sentence_ids[i]
            if idx not in sentence_id2result:
                sentence_id2result[idx] = {
                    "logits": [],
                    "begin_time": -1,
                    "end_time": -1,
                    "text": ""
                }

            sentence_id2result[idx]["logits"].append(y_logits[i])
            sentence_id2result[idx]["begin_time"] = all_begin_times[i]
            sentence_id2result[idx]["end_time"] = all_end_times[i]
            sentence_id2result[idx]["text"] = all_texts[i]

        for idx in sentence_id2result:
            logits = torch.stack(sentence_id2result[idx]["logits"], dim=0)
            logits = torch.mean(logits, dim=0)
            sentence_id2result[idx]["logits"] = logits

            if logits.shape != (3,):
                print("WARNING: No.{} sentence's logits' length is not 3".format(idx))

            sentence_id2result[idx]["pred"] = torch.argmax(logits, dim=0)
            sentence_id2result[idx]["prob"] = F.softmax(logits, dim=0)

        return sentence_id2result


    def predict(self, asr_result_str):
        print("Start detecting interactive.")
        asr_results = json.loads(asr_result_str)["data"]["result"]

        results = {}

        dataset = self.process_asr(asr_results)

        if len(dataset) == 0:
            print("ERROR: input asr is empty.")
            return results

        sentence_id2result = self.inference(dataset)

        for idx in sentence_id2result:
            text = sentence_id2result[idx]["text"]
            begin_time = str(sentence_id2result[idx]["begin_time"])
            end_time = str(sentence_id2result[idx]["end_time"])

            if sentence_id2result[idx]["pred"] == 1:

                if text not in results:
                    results[text] = []

                results[text].append({
                    "begin_time": begin_time,
                    "end_time": end_time,
                    "count": 1,
                    "type": 1,
                })

            elif sentence_id2result[idx]["pred"] == 2:

                if text not in results:
                    results[text] = []

                results[text].append({
                    "begin_time": begin_time,
                    "end_time": end_time,
                    "count": 1,
                    "type": 0,
                })

        print("finish interactive detecting.")
        return json.dumps(results)



interactive_detector = Interactive(model_path=os.path.join(base_path, 'albert'))


