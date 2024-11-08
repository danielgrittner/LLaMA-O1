# Imports and Model Initialization
import contextlib
import copy
import glob
import hashlib
import json
import math
import random
from functools import lru_cache

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, SinkCache,
                          StoppingCriteria, StoppingCriteriaList, Trainer,
                          TrainingArguments)

from grading import check

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

# 数值稳定的 softmax 函数
def robust_softmax(logits):
    logits = torch.tensor(logits) if not isinstance(logits, torch.Tensor) else logits
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return probs, log_probs

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
        return "<positive_rating>"
    elif math.exp(value) < 0.5 and math.exp(value) >= 0:
        return "<negative_rating>"
    else:
        return "<unknow_rating>"


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
            f"<start_of_father_id>{node.parent.index if node.parent else -1}<end_of_father_id><start_of_local_id>{node.index}<end_of_local_id><start_of_thought>{node.state}<end_of_thought><start_of_rating>{value_to_rating_token(node.value)}<end_of_rating>"
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
GENERATE_MAX_NEW_TOKENS = 64
CUT_OFF_LEN = 1024
MAX_CHILDREN_NUM = 5


DummyTransitionProbs = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)


def flatten_tree(node):
    """
    将树结构展开为列表，收集父节点、子节点和对应的值。
    """
    parents = []
    children = []
    values = []
    nodes = [node]
    while nodes:
        current_node = nodes.pop()
        current_idx = meta_action_type_to_index[current_node.meta]
        for child in current_node.children:
            child_idx = meta_action_type_to_index[child.meta]
            parents.append(current_idx)
            children.append(child_idx)
            values.append(np.exp(child.value))
            nodes.append(child)
    return np.array(parents), np.array(children), np.array(values)


def cal_meta_transition_probs(node):
    num_meta_actions = len(meta_action_types)
    # 展开树结构，获取父节点索引、子节点索引和对应的值
    parents, children, values = flatten_tree(node)
    # 初始化转移概率矩阵
    TransitionProbs = np.zeros((num_meta_actions, num_meta_actions))
    # 使用 NumPy 的高级索引和累加来更新矩阵
    if len(parents) > 0:
        np.add.at(TransitionProbs, (parents, children), values)
    return TransitionProbs


def np_softmax(x):
    # 对矩阵的每一行进行 softmax 操作
    max_vals = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_vals)
    sum_e_x = np.sum(e_x, axis=1, keepdims=True)
    return e_x / sum_e_x


@lru_cache()
def sampling_meta_action(node, num=1, TransitionProbs=None):
    if TransitionProbs is None:
        root = get_root(node)
        TransitionProbs = cal_meta_transition_probs(root)
    # 计算转移概率的 softmax
    transition_probs_softmax = np_softmax(TransitionProbs)
    i = meta_action_type_to_index[node.meta]
    p = transition_probs_softmax[i]
    # 进行采样
    meta_actions = np.random.choice(meta_action_types, size=num, p=p)
    return meta_actions


class TreeNode:
    def __init__(self, state, parent=None, index=0):
        self.index = index  # Index of the node in the tree
        self.state = state  # Current state text representation
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of visits
        self.value = 0  # Value estimate of the current node
        self.policy = {}  # Policy probabilities for selecting child nodes
        self.policy_entropy = {}
        self.policy_varentropy = {}
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
        logits = torch.tensor(list(self.policy.values()))
        prob, log_prob = robust_softmax(logits)
        return {key: prob for key, prob in zip(self.policy.keys(), prob)}[child]

    def get_child_policy_entropy(self, child):
        logits = torch.tensor(list(self.policy_entropy.values()))
        prob, log_prob = robust_softmax(logits)
        return {key: prob for key, prob in zip(self.policy_entropy.keys(), prob)}[child]

    def get_child_policy_varentropy(self, child):
        logits = torch.tensor(list(self.policy_varentropy.values()))
        prob, log_prob = robust_softmax(logits)
        return {key: prob for key, prob in zip(self.policy_varentropy.keys(), prob)}[
            child
        ]

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
            "policy": self.policy,
            "policy_entropy": self.policy_entropy,
            "policy_varentropy": self.policy_varentropy,
            "policy_cal_ready_texts": self.policy_cal_ready_texts,
            "value_cal_ready_texts": self.value_cal_ready_texts,
            "true_value_from_tree": self.true_value_from_tree,
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
            node.policy[child_node] = policy_prob
            node.policy_entropy[child_node] = entropy
            node.policy_varentropy[child_node] = varentropy
            node.add_child(child_node)
            child_node.value = self.compute_value(child_node)
            child_node.meta = meta
            # if child_node.meta == "<conclusion>":
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
            (
                child.value
                + self.exploration_const * node.get_child_policy_prob(child)
                # * node.get_child_policy_entropy(child)
                * np.sqrt(total_visits) / (1 + child.visits)
                + self.varentropy_lambda * node.get_child_policy_varentropy(child)
            )
            * random.uniform(0.8, 1.2)
            for child in node.children
        ]
        return node.children[np.argmax(ucb_scores)]

    def identify_leaf(self, node):
        result = set()
        if node.is_leaf() or node.leaf_type in ["successful", "failed"]:
            result.add(node)
        else:
            for child in node.children:
                result |= self.identify_leaf(child)
        return result

    def rectify_values_from_leaf(self, node, value):
        node.rectify_visits += 1

        if not node.true_value_from_tree:
            node.true_value_from_tree = value
        else:
            node.true_value_from_tree += (
                value - node.true_value_from_tree
            ) / node.rectify_visits
        if node.parent:
            self.rectify_values_from_leaf(
                node.parent, node.true_value_from_tree * self.discount_factor
            )