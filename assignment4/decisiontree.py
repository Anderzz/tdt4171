import numpy as np
import pandas as pd
from graphviz import Digraph
import random
import string
import uuid

random.seed(0)
target = '1.5'

class Node:
    """
    Class Node
    """
    def __init__(self, value):
        self.children = []
        self.value = value
        self.id = str(value)
            


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
    try:
        p = examples[tar].value_counts()[1]
    except:
        p = 0
    #p=random.randint(40,60)
    #negative
    try:
        n = examples[tar].value_counts()[2]
    except:
        n = 0
    #n=random.randint(20,39)
    #print(f"pos: {p}, neg: {n}")
    return B(p/(p+n)) - remainder(tar, A, p, n, examples)

def get_id():
    return (''.join(random.choices(string.ascii_lowercase, k=4)))



def learn_decision_tree(examples, attributes, tree=None, parent_examples=()):
    if len(examples) == 1 or len(examples) == 0:
        value = examples[target].unique()[0]
        id = str(uuid.uuid1())
        tree.node(id, label=str(value))
        return plurality_values(parent_examples)


    elif same_classification(examples):
        #return the classification
        classific = examples[target].unique()[0]
        return classific

    elif len(attributes) == 0:
        return plurality_values(examples)
    
    #find the most important attribute
    else:
        dict = {}
        gain = {}
        for col in attributes:
            gain[col] = importance(target, col, examples, rand=True)
        #print(gain)
        
        max_list = sorted(gain.items(), key=lambda x: x[1], reverse=True)
        max_list = list(filter(lambda x: x[0] != '1.5', max_list))
        A = max_list[0][0]
        #print(gain, f"velg {A}")
        id = str(uuid.uuid1())
        tree.node(id, label=A)
        dict_list = {}
        for v in examples[A].unique():
            exs = examples[examples[A]==v]
            new_attrs = attributes.copy().drop(A)
            subtree = learn_decision_tree(exs, new_attrs, tree, examples)
            tree.edge(id, subtree[0], label=str(v))
            dict_list[v] = subtree[1]
    
        return [id, dict]



def plurality_values(e):
    id = str(uuid.uuid1())
    return  [id, e[target].mode()[0]]

def same_classification(e: pd.DataFrame):
    e=e.to_numpy()
    return((e==e[0]).all())






def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    tree = Digraph(name="Decision Tree", filename="dtl")
    print(train)
    res = learn_decision_tree(train, train.columns, tree)
    tree.render(view=True)


if __name__ == "__main__":
    main()
