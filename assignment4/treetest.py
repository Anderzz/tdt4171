from graphviz import Digraph
tree = Digraph(name="Decision Tree", filename="dtl")

#res = learn_decision_tree(train, train.columns, tree)
tree.node(name="attr_1", label="Attr 1?")
tree.node(name="value_1", label="1")
tree.node(name="value_2", label="2")
tree.edge(tail_name="attr_1", head_name="value_1", label="2")
tree.edge(tail_name="attr_1", head_name="value_2", label="1")
tree.render(view=True)