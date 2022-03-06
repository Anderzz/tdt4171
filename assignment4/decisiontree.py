import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphviz import Digraph
import random
import string
from copy import deepcopy

random.seed(0)
target = '1.5'

class Node():
    '''
    Node class, for root and internal nodes. Leaves are values pure
    '''
    def __init__(self, attr, values):
        #super().__init__()
        self.attribute = attr
        self.values = values
        self.children = []
    
    def get_attribute(self):
        return self.attribute

def B(q):
    return 0 if q == 1 or q == 0 else -(q*np.log2(q) + (1-q)*np.log2(1-q))

def remainder(T, A, p, n, examples):
    distinct_val = examples[A].unique()
    splits = []
    for val in distinct_val:
        splits.append(examples[examples[A] == val])

    sum = 0
    for split in splits:
        try:
            pk = split[T].value_counts()[1]
        except:
            pk = 0
        try:
            nk = split[T].value_counts()[2]
        except:
            nk = 0

        sum += ((pk + nk)/(p + n)) * B(pk/(pk + nk))
    return sum

def importance(tar, A, examples, rand=False):
    if rand: return random.uniform(0,1) 
    #positive
    p = examples[tar].value_counts()[1]
    #negative
    n = examples[tar].value_counts()[2]
    return B(p/(p+n)) - remainder(tar, A, p, n, examples)

def get_id():
    return (''.join(random.choices(string.ascii_lowercase, k=4)))

"""class Node:
    def __init__(self, attribute, split=None):
        self.attribute = attribute
        self.children = []
        self.split = split"""

def get_attribute(self):
    return self.attributes

def tree_construct(attr, att_value_list):
    '''
    Trekonstruktør, lager node med attributt og verdier satt.
    '''
    tree_node = Node(attr, att_value_list.get(attr))
    return tree_node

def learn_decision_tree(examples, attributes, graph=None, parent_examples=()):

    attribute_value_pairs = {}
    for att in attributes:
        attribute_value_pairs[att] = examples[att].unique()

    if len(examples) == 0:
        return plurality_values(parent_examples)


    elif same_classification(examples):
        #return the classification
        classific = examples[target].unique()[0]
        return classific

    elif len(attributes) == 0:
        return plurality_values(examples)
    
    #find the most important attribute
    gain = {}
    for col in attributes:
        gain[col] = importance(target, col, examples, rand=True)
    print(gain)
    
    #A = max(gain, key = gain.get)
    max_list = sorted(gain.items(), key=lambda x: x[1], reverse=True)
    A = max_list[1][0] if max_list[0][0] == target else max_list[0][0]
    tree = tree_construct(A,attribute_value_pairs)
    for v in tree.values:#v in examples[A].unique():
        exs = examples[examples[A]==v]
        print(exs)
        #new_attrs = attributes.copy().drop(A)
        #subtree = learn_decision_tree(exs, new_attrs, graph, examples)



def plurality_values(e):
    return  e[target].mode()[0]
    #kanskje prøv e[target].value_counts().idxmax()

def same_classification(e: pd.DataFrame):
    e=e.to_numpy()
    return((e==e[0]).all())






def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    tree = Digraph(name="Decision Tree", filename="dtl")
    tree.node("attr_1", label="a1")
    tree.node("attr_2", label="a2")
    tree.node("attr_3", label="a3")
    tree.edge(tail_name="attr_1", head_name="attr_2")
    tree.edge(tail_name="attr_1", head_name="attr_3")
    #tree.render(view=True)


    learn_decision_tree(train,train.columns,tree)



if __name__ == "__main__":
    main()
