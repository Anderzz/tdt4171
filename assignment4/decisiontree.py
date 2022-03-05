import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphviz import Digraph
import random
import string

#random.seed(0)
target = '1.5'
class Node:
    def __init__(self, children=None, parents=None, right=None, left=None, value=None):
        self.children = []
        self.parents = []
        self.right = right
        self.left = left
        self.value = value

    def print_children(self):
        print(f"Children: {self.children}")



def B(q):
    return 0 if q == 1 or q == 0 else -(q*np.log2(q) + (1-q)*np.log2(1-q))

def remainder(T, A, p, n, examples):
    #An attribute A with d distinct values divides the training set E into subsets E1,...,Ed
    #distinct_val = examples[A].unique()
    distinct_val = [1,2]
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

def importance(tar, A, examples):
    p = examples[tar].value_counts()[1]
    n = examples[tar].value_counts()[2]

    return B(p/(p+n)) - remainder(tar, A, p, n, examples)

def get_id():
    return (''.join(random.choices(string.ascii_lowercase, k=4)))

def learn_decision_tree(examples, attributes, dot=None, parent_examples=()):
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
    attributes = examples.columns
    for col in attributes:
        gain[col] = importance(target, col, examples)
    print(gain)

    A = max(gain, key = gain.get)
    print(A)
    #for v in examples[A].unique():
    #    exs = examples[examples[A]==v]
    #    new_attrs = attributes.copy().drop(A)
    #    subtree = learn_decision_tree(exs, new_attrs, dot, examples)



def plurality_values(e):
    return  e[target].mode()[0]

def same_classification(e: pd.DataFrame):
    e=e.to_numpy()
    return((e==e[0]).all())






def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    learn_decision_tree(train,train.columns)


if __name__ == "__main__":
    main()
