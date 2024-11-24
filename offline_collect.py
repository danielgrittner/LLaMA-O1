# Imports and Model Initialization
import contextlib
import copy
import glob
import hashlib
import json
import math
import random
from functools import lru_cache
import uuid

import accelerate
from llamafactory.api import app
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, SinkCache,
                          StoppingCriteria, StoppingCriteriaList, Trainer,
                          TrainingArguments)

from grading import check, extract_label

import os
import pickle

from datasets import Dataset
from torch.utils.data import DataLoader

import gzip
from datasets import load_dataset
import time



accelerator = accelerate.Accelerator()


def manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




@contextlib.contextmanager
def set_left_padding(tokenizer):
    # Store the original padding side
    original_padding_side = tokenizer.padding_side
    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    # Set padding side to left
    tokenizer.padding_side = "left"
    try:
        yield tokenizer
    finally:
        # Restore original padding side
        tokenizer.padding_side = original_padding_side
        tokenizer.truncation_side = original_truncation_side


@contextlib.contextmanager
def set_left_truncate(tokenizer):
    # Store the original padding side
    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        yield tokenizer
    finally:
        tokenizer.truncation_side = original_truncation_side


def value_to_rating_token(value):
    if math.exp(value) >= 0.5 and math.exp(value) <= 1:
        return "True"
    elif math.exp(value) < 0.5 and math.exp(value) >= 0:
        return "False"
    else:
        return "None"


def tree_to_string(node):
    cur = f"<start_of_father_id>{node.parent.index if node.parent else -1}<end_of_father_id><start_of_local_id>{node.index}<end_of_local_id><start_of_thought>{node.state}<end_of_thought><start_of_rating>{value_to_rating_token(node.value)}<end_of_rating>"
    childs_strings = "\n".join([tree_to_string(child) for child in node.children])
    return cur + "\n" + childs_strings


def path_to_string(node):
    path = []
    while node:
        path.append(node)
        node = node.parent
    string = "\n".join(
        [
            node.state
            for node in path[::-1]
        ]
    )
    return string


def get_max_node_id_in_tree(node):
    if not node.parent:
        while node.parent:
            node = node.parent
    max_id = node.index
    for child in node.children:
        max_id = max(max_id, get_max_node_id_in_tree(child))
    return max_id


def get_root(node):
    while node.parent:
        node = node.parent
    return node


select_prefix = ""
meta_action_types = ["<expansion>", "<problem>", "<critic>", "<refine>", "<conclusion>"]

meta_action_type_to_index = {meta: i for i, meta in enumerate(meta_action_types)}


LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E
GENERATE_MAX_NEW_TOKENS = 1024
CUT_OFF_LEN = 40960
MAX_CHILDREN_NUM = 4


@lru_cache()
def sampling_meta_action(node, num=1, TransitionProbs=None):
    if node.meta == "<conclusion>":
        return ["<critic>"] * num
    if node.meta == "<critic>":
        return ["<refine>"] * num
    if 'answer' in node.state.lower():
        child_metas = [child.meta for child in node.children]
        child_leaf_types = [child.leaf_type for child in node.children]
        value = math.exp(node.value)
        if '<conclusion>' in child_metas and 'successful' in child_leaf_types:
            return ['<critic>'] * num
        return random.choices(['<conclusion>','<critic>'], [value,1-value],k=num)
    if node.meta == "<problem>":
        return ["<expansion>"] * num
    if node.meta == "<expansion>":
        value = math.exp(node.value)
        # if value < 0.1:
        #     return ['<problem>']  * num
        # elif 0.2 <= value <= 0.6:
        #     return ['<critic>'] * num
        # else:
        #     return ['<expansion>'] * num
        return random.choices(['<problem>', '<critic>','<expansion>'], [(1-value)*0.2,(1-value)*0.8,value],k=num)
    if node.meta == "<refine>":
        return ["<expansion>"] * num
    
    return ["<expansion>"] * num

class TreeNode:
    def __init__(self, state, parent=None, index=0):
        self.index = index  # Index of the node in the tree
        self.state = state  # Current state text representation
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of visits
        self.value = 0  # Value estimate of the current node
        self.policy_probs = []  # Policy probabilities for selecting child nodes
        self.policy_entropy = []
        self.policy_varentropy = []
        self.policy_cal_ready_texts = ""
        self.value_cal_ready_texts = ""
        self.true_value_from_tree = None
        self.leaf_type = ""
        self.rectify_visits = 0
        self.original_value = 0
        self.meta = "<problem>"

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_leaf(self):
        return len(self.children) == 0

    def get_path_reward(self):
        path_len = 1
        reward = 0
        node = self
        while node.parent:
            path_len += 1
            reward += node.value
            node = node.parent
        return reward / path_len

    def should_expand(self):
        if len(self.children) == 0:
            return True
        if (
            len(self.children) < MAX_CHILDREN_NUM
        ):  # max([child.value for child in self.children]) < self.value or
            return True
        return False

    def get_child_policy_prob(self, child):
        logits = torch.tensor(self.policy_probs)
        prob, log_prob = robust_softmax(logits)
        return prob[self.children.index(child)]

    def get_child_policy_entropy(self, child):
        logits = torch.tensor(self.policy_entropy)
        prob, log_prob = robust_softmax(logits)
        return prob[self.children.index(child)]

    def get_child_policy_varentropy(self, child):
        logits = torch.tensor(self.policy_varentropy)
        prob, log_prob = robust_softmax(logits)
        return prob[self.children.index(child)]

    def to_dict(self):
        """Convert TreeNode and its subtree to a dictionary for serialization."""
        # Recursive function to convert the entire tree to a dict
        node_dict = {
            "index": self.index,
            "state": self.state,
            "parent": (
                self.parent.index if self.parent else None
            ),  # Store parent index or None
            "visits": self.visits,
            "value": self.value,
            "policy_probs": self.policy_probs,
            "policy_entropy": self.policy_entropy,
            "policy_varentropy": self.policy_varentropy,
            "policy_cal_ready_texts": self.policy_cal_ready_texts,
            "value_cal_ready_texts": self.value_cal_ready_texts,
            "true_value_from_tree": self.true_value_from_tree if self.true_value_from_tree is not None else None,
            "leaf_type": self.leaf_type,
            "rectify_visits": self.rectify_visits,
            "original_value": self.original_value,
            "meta": self.meta,
            "children": [
                child.to_dict() for child in self.children
            ],  # Recursively add children
        }
        return node_dict

    @staticmethod
    def from_dict(node_dict):
        """Rebuild the TreeNode from a dictionary (recursive)."""
        node = TreeNode(
            state=node_dict["state"],
            index=node_dict["index"],
            parent=None,  # Parent will be set later
        )
        node.visits = node_dict["visits"]
        node.value = node_dict["value"]
        node.policy = node_dict["policy"]
        node.policy_entropy = node_dict["policy_entropy"]
        node.policy_varentropy = node_dict["policy_varentropy"]
        node.policy_cal_ready_texts = node_dict["policy_cal_ready_texts"]
        node.value_cal_ready_texts = node_dict["value_cal_ready_texts"]
        node.true_value_from_tree = node_dict["true_value_from_tree"]
        node.leaf_type = node_dict["leaf_type"]
        node.rectify_visits = node_dict["rectify_visits"]
        node.original_value = node_dict["original_value"]
        node.meta = node_dict["meta"]

        # Recursively add children
        for child_dict in node_dict["children"]:
            child_node = TreeNode.from_dict(child_dict)
            child_node.parent = node  # Set parent for the child
            node.add_child(child_node)

        return node

    def save_to_json(self, filepath):
        """Save the entire tree structure (root) to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def load_from_json(filepath):
        """Load the tree structure from a JSON file."""
        with open(filepath, "r") as f:
            node_dict = json.load(f)
        return TreeNode.from_dict(node_dict)


# MCTS Search
class MCTS:
    def __init__(
        self,
        envoirment,
        model,
        tokenizer,
        num_simulations=-1,
        num_candidates_per_expansion=2,
        exploration_const=1.414,
        discount_factor=0.9,
        reward_epsilon=1e-6,
        patient=2,
    ):
        self.envoirment = envoirment
        self.model = model
        self.tokenizer = tokenizer
        self.num_simulations = num_simulations if num_simulations != -1 else 32
        self.exploration_const = exploration_const
        self.patient = patient
        self.discount_factor = discount_factor
        self.num_candidates = num_candidates_per_expansion
        self.reward_epsilon = reward_epsilon
        self.varentropy_lambda = 0.1

    def search(self, root_node):
        if not root_node.children:
            root_node.value = 0

        for _ in tqdm(range(self.num_simulations)):
            self.simulate(root_node)
            if self.patient <= 0:
                break

        for leaf in self.identify_leaf(root_node):
            if leaf.leaf_type == "successful":
                self.rectify_values_from_leaf(leaf, 0)
            else:
                self.rectify_values_from_leaf(leaf, np.log(self.reward_epsilon))

        return root_node

        # return self.get_policy_from_visits(root_node)

    def simulate(self, node):
        if node.is_leaf() or node.should_expand():
            value = self.expand_node(node) * self.discount_factor
        else:
            best_child = self.select_action(node)
            value = self.simulate(best_child) * self.discount_factor
        node.visits += 1
        node.value += (value - node.value) / node.visits
        return node.value

    def expand_node(self, node):
        texts, policy_probs, entropys, varentropys, metas = meta_compute_policy_head(
            self.model,
            self.tokenizer,
            node,
            self.num_candidates,
            envoirment=self.envoirment,
        )

        for i, (text, policy_prob, entropy, varentropy, meta) in enumerate(
            zip(texts, policy_probs, entropys, varentropys, metas)
        ):
            child_node = TreeNode(
                state=text, parent=node, index=get_max_node_id_in_tree(node) + 1
            )
            # child_node.policy = policy_probs[i]
            node.policy_probs.append(policy_prob)
            node.policy_entropy.append(entropy)
            node.policy_varentropy.append(varentropy)
            node.add_child(child_node)
            child_node.value = self.compute_value(child_node)
            child_node.meta = meta
            if child_node.meta == "<conclusion>":
                orm = self.envoirment.compute_rule_orm_head(child_node)
                if orm == True:
                    self.patient -= 1
                    child_node.leaf_type = "successful"
                elif orm == False:
                    child_node.leaf_type = "failed"
            print(
                f"Id:{node.index}->{child_node.index}, Child: {text}, Policy: {node.get_child_policy_prob(child_node)}, Value: {math.exp(child_node.value)}"
            )
        return self.select_action(node).value

    def compute_value(self, node):
        # Use the model to predict the value of the current state
        value = compute_value_head(self.model, self.tokenizer, node)
        node.value = value
        node.original_value = copy.deepcopy(value)
        return value

    def select_action(self, node):
        total_visits = sum(child.visits for child in node.children)
        ucb_scores = [
                child.value
                + self.exploration_const * node.get_child_policy_prob(child)
                # * node.get_child_policy_entropy(child)
                * np.sqrt(total_visits) / (1 + child.visits)
                + self.varentropy_lambda * node.get_child_policy_varentropy(child)
            for child in node.children
        ]
        return node.children[np.argmax(ucb_scores)]

    def identify_leaf(self, node):
        result = set()
        if node.is_leaf() and node.leaf_type in ["successful", "failed"]:
            result.add(node)
        else:
            for child in node.children:
                result |= self.identify_leaf(child)
        return result

    def rectify_values_from_leaf(self, node, value):
        node.rectify_visits += 1

        if not node.true_value_from_tree:
            node.true_value_from_tree = float(value)
        else:
            node.true_value_from_tree += float((
                value - node.true_value_from_tree
            ) / node.rectify_visits)
        if node.parent:
            self.rectify_values_from_leaf(
                node.parent, node.true_value_from_tree * self.discount_factor
            )

hint_for_expansion = "Try complete this solution step-by-step that can got final answer."

hint_for_critics = (
    "Point out flaws or give hints to above content step-by-step. \n hint: true answer is {}"
)
hint_for_refine = (
    "Try solute this problem step-by-step that can got final answer in better alternative ways. You can refer to the above content and critiques."
)
hint_for_conclusion = "Try to summarize above contents and draw a conclusion. Final answer should bracket in \\box{answer}"
hint_for_divide_and_conquer = "Try divide the problem \n{}\n into smaller easier sub-problems and solve them divide-and-conquer.\n hint: true answer is {}"

# 模板生成函数
def problem_declaration_template(problem):
    return f"<problem>{problem}"


def selection_head_template(tree):
    return tree.to_string() + "\n<start_of_father_id>"


def policy_head_template(selected_node, local_id, meta="", hint=""):
    context = path_to_string(selected_node)
    return context, hint


def value_head_template(selected_node):
    context = path_to_string(selected_node.parent) + f"\n<start_of_father_id>{selected_node.parent.index if selected_node.parent else -1}<end_of_father_id><start_of_local_id>{selected_node.index}<end_of_local_id><start_of_thought>{selected_node.state}<end_of_thought>"
    hint = 'Try to evaluate the current solution. True or False?'
    return context, hint

selection_head_stopping_criteria = ["<end_of_father_id>"]

policy_head_stopping_criteria = ["<end_of_thought>",'<start_of_rating>','<expansion>','<problem>','<critic>','<refine>','<conclusion>']

value_head_stopping_criteria = ["<end_of_rating>"]


def clean_generated_text(text):
    text = text.replace("<problem>", "").replace("<critic>", "").replace("<refine>", "").replace("<conclusion>", "")
    return text[: text.find("<end_of_thought>")]


def find_max_reward_path(node):
    path = 0
    reward = 0
    while node:
        reward += node.value
        path += 1
        if not node.children:
            break
        node = max(node.children, key=lambda x: x.value)
    return math.exp(reward), path


# 数值稳定的 softmax 函数
def robust_softmax(logits):
    logits = torch.tensor(logits) if not isinstance(logits, torch.Tensor) else logits
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return probs, log_probs


def length_normed_log_probs(
    sequence_ids,
    logits_tensor,
    attention_mask=None,
    return_entropy=False,
    return_varentropy=False,
):
    logits_tensor = logits_tensor[..., :-1, :].contiguous()
    sequence_ids = sequence_ids[..., 1:].contiguous()
    attention_mask = (
        attention_mask[..., 1:].contiguous() if attention_mask is not None else None
    )
    log_probs = F.log_softmax(logits_tensor, dim=-1)
    selected_log_probs = log_probs.gather(2, sequence_ids.unsqueeze(-1)).squeeze(-1)

    if attention_mask is not None:
        selected_log_probs = selected_log_probs * attention_mask

    summed_log_probs = selected_log_probs.sum(dim=1)
    length = (
        sequence_ids.size(1) if attention_mask is None else attention_mask.sum(dim=1)
    )

    # Check for length == 0 to avoid division by zero
    normalized_log_probs = torch.where(
        length > 0, summed_log_probs / length, torch.zeros_like(summed_log_probs)
    )

    if return_entropy or return_varentropy:
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        if attention_mask is not None:
            entropy = entropy * attention_mask
        summed_entropy = entropy.sum(dim=1)
        normalized_entropy = torch.where(
            length > 0, summed_entropy / length, torch.zeros_like(summed_entropy)
        )

    if return_varentropy:
        varentropy = torch.sum(probs * (log_probs + entropy.unsqueeze(-1)) ** 2, dim=-1)
        if attention_mask is not None:
            varentropy = varentropy * attention_mask
        summed_varentropy = varentropy.sum(dim=1)
        normalized_varentropy = torch.where(
            length > 0, summed_varentropy / length, torch.zeros_like(summed_varentropy)
        )
        return normalized_log_probs, normalized_entropy, normalized_varentropy

    if return_entropy:
        return normalized_log_probs, normalized_entropy
    else:
        return normalized_log_probs



def apply_chat_template(tokenizer, history):
    history = [{'role':['user','assistant'][i % 2],'content':j} for i,j in enumerate(['hi',] + history)]
    return tokenizer.apply_chat_template(history,tokenize=False)

# 策略生成的主要函数
@torch.no_grad()
def compute_policy_head(
    model, tokenizer, selected_node, num_candidates=3, meta="", envoirment=None
):
    local_id = get_max_node_id_in_tree(selected_node) + 1
    hint_text = {
        "<conclusion>": hint_for_conclusion,
        "<problem>": hint_for_divide_and_conquer.format(get_root(selected_node).state,envoirment.get_ground_truth(selected_node)),
        "<critic>": hint_for_critics.format(envoirment.get_ground_truth(selected_node)),
        "<refine>": hint_for_refine,
        "<expansion>": hint_for_expansion
    }.get(meta)


    contex, hint = policy_head_template(selected_node, local_id, meta, hint_text)#
    with set_left_truncate(tokenizer):
        inputs_string = apply_chat_template(tokenizer, [contex, hint])
        inputs = tokenizer(
            inputs_string,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=CUT_OFF_LEN,
        )
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

    outputs = accelerator.unwrap_model(model).generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=GENERATE_MAX_NEW_TOKENS,
        do_sample=True,
        num_return_sequences=num_candidates,
        return_dict_in_generate=True,
        output_scores=True,
        # temperature=1.5,
        output_logits=True,
        stop_strings=policy_head_stopping_criteria,
        tokenizer=tokenizer,
    )

    generated_sequences = outputs.sequences[:, inputs["input_ids"].size(1) :]
    generated_sequences_mask = generated_sequences != tokenizer.pad_token_id
    generated_texts = tokenizer.batch_decode(
        generated_sequences, skip_special_tokens=True
    )

    logits = torch.stack(outputs.logits, dim=1)
    normalized_log_probs, normalized_entropy, varentropy = length_normed_log_probs(
        generated_sequences,
        logits,
        attention_mask=generated_sequences_mask,
        return_entropy=True,
        return_varentropy=True,
    )

    normalized_probs = torch.exp(normalized_log_probs)

    generated_texts = [clean_generated_text(text) for text in generated_texts]
    for i, generated_text in enumerate(generated_texts):
        generated_texts[i] = generated_text
        if 'The final answer is' in generated_text:
            meta = "<conclusion>"

    return (
        generated_texts,
        normalized_probs.tolist(),
        normalized_entropy.tolist(),
        varentropy.tolist(),
        [
            meta,
        ]
        * num_candidates,
    )


# 价值头生成函数
@torch.no_grad()
def compute_value_head(model, tokenizer, node):
    text_for_value, hint = value_head_template(node)
    with set_left_truncate(tokenizer):
        text_for_value = apply_chat_template(tokenizer, [text_for_value, hint, 'True'])
        inputs = tokenizer(
            text_for_value,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=CUT_OFF_LEN,
        )
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits

    last_logits = logits[:, -2, :]
    positive_token_id = tokenizer.convert_tokens_to_ids("True")
    negative_token_id = tokenizer.convert_tokens_to_ids("False")

    positive_logit = last_logits[:, positive_token_id]
    negative_logit = last_logits[:, negative_token_id]
    value_logits = torch.stack([positive_logit, negative_logit], dim=1)

    probs, log_probs = robust_softmax(value_logits)
    return log_probs[:, 0].item()


# 元策略生成函数
@torch.no_grad()
def meta_compute_policy_head(
    model, tokenizer, selected_node, num_candidates=3, meta_ratio=0.5, envoirment=None
):
    metas = sampling_meta_action(selected_node, num_candidates)
    generated_texts, policy_probs, normalized_entropys, varentropys = [], [], [], []

    for meta in metas:
        texts, policy_probs, normalized_entropy, varentropy, _ = compute_policy_head(
            model,
            tokenizer,
            selected_node,
            num_candidates=1,
            meta=meta,
            envoirment=envoirment,
        )
        generated_texts.append(texts[0])
        policy_probs.append(policy_probs[0])
        normalized_entropys.append(normalized_entropy[0])
        varentropys.append(varentropy[0])

    return generated_texts, policy_probs, normalized_entropys, varentropys, metas


def padding_nodes(tensor, max_len):
    feature_dim = tensor.size(-1)
    pad_len = max_len - tensor.size(1)
    pad_tensor = torch.zeros(
        tensor.size(0), pad_len, feature_dim, dtype=tensor.dtype, device=tensor.device
    )
    return torch.cat([tensor, pad_tensor], dim=1)

def prompt_policy_predict(nodes):
    text_for_policys = [
        policy_head_template(node.parent, node.index) + node.state for node in nodes
    ]
    actions = [node.state for node in nodes]
    return text_for_policys, actions

def tokenize_policy_predict(text_for_policys, targets, tokenizer):
    with set_left_truncate(tokenizer):
        # with set_left_padding(tokenizer):
        inputs = tokenizer(
            text_for_policys,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=CUT_OFF_LEN,
        )
        target = tokenizer(
            targets,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=CUT_OFF_LEN,
        )
    ret = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "action_input_ids": target["input_ids"],
        "action_attention_mask": target["attention_mask"],
    }
    return ret


def forward_policy_predict(model, tokenizer, inputs):
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    target_ids = inputs["target"]
    target_mask = inputs["target_attention_mask"]
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    )
    logits = outputs.logits[:, :-1, :][:, -target_ids[:, 1:].shape[-1] :]
    log_probs = F.log_softmax(logits, dim=-1)
    seleted_log_probs = log_probs.gather(2, target_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return seleted_log_probs

def prompt_value_predict(nodes):
    text_for_values = [value_head_template(node) for node in nodes]
    return text_for_values

def tokenize_value_predict(text_for_values, tokenizer):
    with set_left_truncate(tokenizer):
        
        inputs = tokenizer(
            text_for_values,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=CUT_OFF_LEN,
        )
    inputs = {"value_" + k: v for k, v in inputs.items()}
    return inputs



def forward_value_predict(model, tokenizer, inputs):
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    input_ids = inputs.pop("value_input_ids")
    attention_mask = inputs.pop("value_attention_mask")
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    )
    logits = outputs.logits
    pos = attention_mask.sum(dim=1) - 1  # [batch_size]

    # 获取 "True" 和 "False" 的 token ID
    positive_token_id = tokenizer.convert_tokens_to_ids("True")
    negative_token_id = tokenizer.convert_tokens_to_ids("False")

    # 构建索引张量
    batch_size = logits.size(0)
    indices = torch.tensor(
        [positive_token_id, negative_token_id], device=accelerator.device
    )  # [2]

    # 扩展 indices 以匹配输入 logits 的维度
    selected_logit = logits[range(batch_size), pos]  # [batch_size, num_tokens]
    selected_logit = selected_logit[:, indices]  # 提取每行中指定 token 的 logits

    return selected_logit


def get_path_reward_real(node):
    path_len = 1
    reward = 0
    while node.parent:
        path_len += 1
        reward += (
            node.true_value_from_tree
            if node.true_value_from_tree is not None
            else node.value
        )
        node = node.parent
    return reward / path_len


def get_path_reward_sim(node):
    path_len = 1
    reward = 0
    while node.parent:
        path_len += 1
        reward += node.original_value
        node = node.parent
    return reward / path_len


def traverse_tree(node):
    """
    Generator to traverse the entire tree from the root node
    """
    visited = set()
    nodes = [node]
    while nodes:
        current_node = nodes.pop()
        if current_node not in visited:
            visited.add(current_node)
            yield current_node
            nodes.extend(current_node.children)
        else:
            continue


def compute_gae_from_node(node, gamma=0.99, lambda_=0.95):
    # 回溯到根节点并记录路径
    path = []
    current_node = node
    while current_node.parent is not None:
        path.append(current_node)
        current_node = current_node.parent

    # 从根节点（路径起点）向下遍历到目标节点，逐步计算 GAE
    gae = 0
    factor = 1  # 用于累乘 (gamma * lambda) 的系数

    # 从根节点开始遍历路径到指定节点
    for i in range(len(path) - 1):  # path[-1] 是目标节点，不需要再计算 TD 误差
        current_node = path[i]
        next_node = path[i + 1]
        next_node_reward = (
            next_node.true_value_from_tree
            if next_node.true_value_from_tree is not None
            else next_node.value
        )
        next_node_value = next_node.value
        current_node_value = current_node.value

        # 计算 TD 误差
        td_error = next_node_reward + gamma * next_node_value - current_node_value
        # 根据 GAE 累积 TD 误差
        gae += factor * td_error
        # 更新系数，准备下一步的累积
        factor *= gamma * lambda_

    return gae

def collator_fn(batch):
    batch = {
        k: pad_sequence(
            [torch.tensor(example[k]).squeeze().unsqueeze(-1) for example in batch],
            True,
            0,
        ).squeeze()
        for k in batch[0].keys()
        if k not in ["indices", "weights"]
    }
    return batch


def get_md5(string):
    return hashlib.md5(str(string).encode()).hexdigest()

class RLSPDataCollector:
    def __init__(self, environment, model, tokenizer, mcts, patient=2, replay_buffer_path='./buffer/'):
        self.environment = environment
        self.model = model
        self.mcts = mcts
        self.tokenizer = tokenizer
        self.replay_buffer_path = replay_buffer_path
        os.makedirs(replay_buffer_path, exist_ok=True)
        self.patient = patient

    def self_play(self, initial_state):
        """Perform self-play to generate experiences."""
        self.mcts.patient = self.patient
        self.model.eval()
        root_node = TreeNode(state=initial_state)
        root_node.meta = "<problem>"
        root_node = self.mcts.search(root_node)
        root_node.save_to_json(self.replay_buffer_path + f"{get_md5(initial_state)}_{time.time()}.json")
        return root_node

    def collect_experience_from_node(self, root_node):
        """Traverse the MCTS tree to collect experiences and store them in the replay buffer."""
        experiences = []
        for node in traverse_tree(root_node):
            if node == root_node:
                continue
            reward = (
                node.true_value_from_tree
                if node.true_value_from_tree is not None
                else node.value
            )
            advantage = compute_gae_from_node(node)

            text_for_policys, text_for_actions = prompt_policy_predict([node])
            advantage_tensor = torch.tensor([advantage], dtype=torch.float32).unsqueeze(
                0
            )
            text_for_values = prompt_value_predict([node], self.tokenizer)
            value_target = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)

            # Store the experience with initial priority
            experience = {
                "advantage": advantage_tensor,
                "value_target": value_target,
                "text_for_policy":text_for_policys[0], 
                "action":text_for_actions[0],
                "value_input": text_for_values[0],
            }
            experiences.append(experience)

        return experiences

    def create_dataset_from_buffer(self, batch_size, times, beta=0.4):
        """Sample a batch from the replay buffer using PER."""
        nodes_files = glob.glob(self.replay_buffer_path + "*.json")
        
        experiences = []
        
        for node_file in nodes_files:
            node = TreeNode.load_from_json(node_file)
            experiences.extend(self.collect_experience_from_node(node))

        if len(experiences) == 0:
            return None
        
        data = {
            "advantage": [],
            "value_target": [],
            "input_ids": [],
            "attention_mask": [],
            "target": [],
            "target_attention_mask": [],
            "value_input_ids": [],
            "value_attention_mask": [],
        }

        for sample in experiences:
            data["advantage"].append(sample.get("advantage", 0))
            data["value_target"].append(sample.get("value_target", 0))
            
            policy_ret = tokenize_policy_predict([sample.get("text_for_policy")], [sample.get("action")], self.tokenizer)
            data["input_ids"].append(policy_ret.get("input_ids", 0))
            data["attention_mask"].append(policy_ret.get("attention_mask", 0))
            data["action_input_ids"].append(policy_ret.get("action_input_ids", 0))
            data["action_attention_mask"].append(
                policy_ret.get("action_attention_mask", 0)
            )

            value_ret = tokenize_value_predict([sample.get("value_input")], self.tokenizer)
            data["value_input_ids"].append(value_ret.get("value_input_ids", 0))
            data["value_attention_mask"].append(
                value_ret.get("value_attention_mask", 0)
            )

        dataset = Dataset.from_dict(data)
        return dataset

class Environment:
    def __init__(self, problems):
        """
        初始化环境。

        参数：
        - problems: 一个包含数学问题和答案的字典列表，每个字典包含 'problem' 和 'ground_truth' 键。
        """
        self.problems = problems
        self.num_problems = len(problems)
        self.inverse_mapping = {
            problem["problem"]: problem["ground_truth"]
            for problem in problems
        }

    def sample_initial_state(self):
        """
        从问题列表中随机采样一个初始状态（数学问题）。

        返回：
        - initial_state: 选中的问题文本。
        - ground_truth: 该问题的正确答案，用于后续的答案验证。
        """
        selected_problem = random.choice(self.problems)
        initial_state = selected_problem["problem"]
        ground_truth = selected_problem["ground_truth"]
        return initial_state, ground_truth

    def is_terminal_state(self, state, ground_truth):
        """
        判断当前状态是否为终止状态（正确答案）。

        参数：
        - state: 当前状态文本。
        - ground_truth: 当前问题的正确答案。

        返回：
        - is_terminal: 布尔值，表示是否为终止状态。
        """
        # 使用 compute_rule_orm_head 函数判断
        result = self.compute_rule_orm_head(state, ground_truth)
        return result

    def get_ground_truth(self, node):
        return self.inverse_mapping.get(get_root(node).state)

    # 判断状态是否为正确答案的函数
    def compute_rule_orm_head(self, node):
        """
        使用 grading 模块的 check 函数判断状态是否为正确答案。

        参数：
        - state: 当前状态文本。
        - ground_truth: 当前问题的正确答案。

        返回：
        - result: 布尔值，表示状态是否为正确答案。
        """
        # 将状态和正确答案传入 check 函数进行比较
        try:
            ground_truth = self.inverse_mapping.get(get_root(node).state)
            result = check(ground_truth, node.state, "")
            return result
        except:
            return None
        
    def get_ground_truth_label(self, node):
        return extract_label(self.inverse_mapping.get(get_root(node).state),"")



# 假设您已经定义了 TreeNode、MCTS 和 RLSPTrainer 类

# 加载模型和 tokenizer
model_name =  "google/gemma-2-2b-it" # "/mnt/hwfile/ai4chem/CKPT/longcot_pt_GEMMA_ZD_10_23_1" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, use_cache=True
)

if model_name.lower().find("llama") != -1:
    tokenizer.pad_token = tokenizer.eos_token

# # 设置 LoRA 配置
# lora_config = LoraConfig(
#     r=32,  # 低秩矩阵的秩
#     lora_alpha=16,  # LoRA 的缩放系数
#     target_modules=[
#         "k_proj",
#         "q_proj",
#         "o_proj",
#         "v_proj",
#         "down_proj",
#         "gate_proj",
#         "up_proj",
#     ],  # 目标模块，通常是查询和键的投影层
#     lora_dropout=0.1,  # dropout 概率
#     bias="none",  # 不在 LoRA 中包含偏置
# )

# # 使用 peft 将模型转换为 LoRA 微调模型
# model = get_peft_model(model, lora_config)

print("Model successfully converted to LoRA format.")

# # 初始化优化器
# optimizer = AdamW(model.parameters(), lr=1e-4)


# 初始状态和 MCTS 参数
num_simulations = 1024
num_candidates_per_expansion = 1
exploration_const = 1.4
discount_factor = 0.9
reward_epsilon = 1e-6
patient = 1



ds0 = load_dataset("openai/gsm8k", "main")["train"]
ds = load_dataset("lighteval/MATH", "all")['train']
ds0 = ds0.shuffle(int(uuid.uuid4()) % (2**32 - 1))
ds = ds.shuffle(int(uuid.uuid4()) % (2**32 - 1))

manual_seed(int(uuid.uuid4())  % (2**32 - 1))

problems0 = [{"problem": p["question"], "ground_truth": p["answer"]} for p in ds0]



problems = [{"problem": p['problem'], "ground_truth": p['solution']} for p in ds]

problems = problems0 + problems

envoirment = Environment(problems)

# 创建 MCTS 实例
mcts = MCTS(
    envoirment=envoirment,
    model=model,
    tokenizer=tokenizer,
    num_simulations=num_simulations,
    num_candidates_per_expansion=num_candidates_per_expansion,
    exploration_const=exploration_const,
    discount_factor=discount_factor,
    reward_epsilon=reward_epsilon,
    patient=patient,
)

model = model.to(accelerator.device)

collector = RLSPDataCollector(envoirment, model, tokenizer, mcts, patient=patient)

for _ in range(1000):
    initial_state, ground_truth = envoirment.sample_initial_state()
    root_node = collector.self_play(initial_state)