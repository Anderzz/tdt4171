import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphviz import Digraph
import random
import string

random.seed(0)

class Node:
    def __init__(self, children=None, parents=None, right=None, left=None, value=None):
        self.children = []
        self.parents = []
        self.right = right
        self.left = left
        self.value = value

    def print_children(self):
        print(f"Children: {self.children}")


target = '1.5'

def B(q):
    return 0 if q == 1 or q == 0 else -(q*np.log2(q) + (1-q)*np.log2(1-q))

def remainder(tar, attr, p, n, examples):
    distinct_values = examples[tar].unique()
    splits = []
    for dist_val in distinct_values:
        splits.append(examples[examples[attr] == dist_val])

    sum = 0
    for split in splits:
        try:
            pk = split[tar].value_counts()[0]
        except:
            pk = 0
        try:
            nk = split[tar].value_counts()[1]
        except:
            nk = 0

        sum += ((pk + nk)/(p + n)) * B(pk/(pk + nk))
    return sum

def importance(tar, attrs, examples, rand=False):
    if rand:
        return random.random()
    #positive examples
    p = examples[tar].value_counts().values[0]
    #negative examples
    n = examples[tar].value_counts().values[1]
    return B(p/(p+n)) - remainder(tar, attrs, p, n, examples)

def get_id():
    return (''.join(random.choices(string.ascii_lowercase, k=4)))

def learn_decision_tree(examples, attributes, dot=None, parent_examples=()):
    if len(examples) == 0:
        return plurality_values(parent_examples)
    elif same_classification(examples):
        #return the classification
        classific = examples[target].unique()[0]
        dot.node(get_id(), str(classific))
        return [get_id(), classific]
    elif len(attributes) == 0:
        return plurality_values(examples)
    
    #find the most important attribute
    gain = {}
    dict = {}
    attributes = examples.columns
    for col in attributes:
        gain[col] = importance(target, col, examples, rand=True)
    #remove the column '1.5'
    try: del gain[target]
    except: pass
    A = max(gain, key = gain.get)
    ide = str(id(A))
    dot.node(ide, label = A)
    dict_list = {}
    for v in examples[A].unique():
        exs = examples[examples[A]==v]
        new_attrs = attributes.copy().drop(A)
        subtree = learn_decision_tree(exs, new_attrs, dot, examples)
        dot.edge(ide, subtree[0], label = str(v))
        dict_list[v] = subtree[1]
    dict[A] = dict_list
    return [ide, dict]



def plurality_values(e):
    return [get_id(), e[target].mode()[0]]

def same_classification(e: pd.DataFrame):
    e=e.to_numpy()
    return((e==e[0]).all())




train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
df = pd.DataFrame(train)
gain = {}
attributes = train.columns

graph = Digraph()

def main():
    dot = Digraph('graf', filename='graf_out.gv')
    #res = importance(target, attributes, train)
    res=learn_decision_tree(train,attributes,dot,None)
    print(res)
    graph.view()

if __name__ == "__main__":
    main()
