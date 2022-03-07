from msilib.schema import Class
import numpy as np
import pandas as pd
from graphviz import Digraph
import random
import string

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
    #p. 1222 in the book
    #returns the entropy
    return 0 if q == 1 or q == 0 else -(q*np.log2(q) + (1-q)*np.log2(1-q))

def remainder(T, A, p, n, examples):
    #p 1223 in the book
    #returns the expected remaining entropy after testing A
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
    #p. 1223 in the book
    #the information gain from the attribute test on A is the expected reduction in entropy
    if rand: return random.uniform(0,1) 
    #positive
    try:
        p = examples[tar].value_counts()[1]
    except:
        p = 0
    #negative
    try:
        n = examples[tar].value_counts()[2]
    except:
        n = 0

    return B(p/(p+n)) - remainder(tar, A, p, n, examples)

def get_id():
    #make a random id
    return (''.join(random.choices(string.ascii_lowercase, k=10)))


def plurality_values(e, tree=None):
    #return the most common value
    value = e[target].mode()[0]
    id = get_id()
    tree.node(id, label = str(value))
    return [id, str(value)]

def same_classification(e: pd.Series, tree: Digraph):
    #return the classification
    classificaion = e.unique()[0]
    id = get_id()
    tree.node(id, label = str(classificaion))
    return [id, str(classificaion)]

def learn_decision_tree(examples, attributes, tree=None, parent_examples=()):

    if examples.empty:
        return plurality_values(parent_examples,tree)

    elif len(examples[target].unique()) == 1:
        return same_classification(examples[target], tree)

    elif attributes.empty:
        return plurality_values(examples,tree)
    
    #find the most important attribute
    else:
        dict = {}
        gain = {}
        for col in attributes:
            gain[col] = importance(target, col, examples, rand=False)
        
        max_list = sorted(gain.items(), key=lambda x: x[1], reverse=True)
        max_list = list(filter(lambda x: x[0] != '1.5', max_list))
        A = max_list[0][0]
        id = get_id()
        tree.node(id, label=A)
        dict_list = {}
        for v in examples[A].unique():
            exs = examples[examples[A]==v]
            new_attrs = attributes.copy().drop(A)
            subtree = learn_decision_tree(exs, new_attrs, tree, examples)
            tree.edge(id, subtree[0], label=str(v))
            dict_list[v] = subtree[1]
        
        dict[A] = dict_list
        return [id, dict]





def main():
    train = pd.read_csv("train.csv")
    train.rename(columns={'1': 'a1', '2': 'a2'}, inplace=True)
    print(train)

    test = pd.read_csv("test.csv")
    tree = Digraph(name="Decision Tree", filename="dtl")
    res = learn_decision_tree(train, train.columns, tree)
    #print(res[1])
    tree.render(view=True)
    #print(train[target].unique()[0])
    #print(value)


if __name__ == "__main__":
    main()
