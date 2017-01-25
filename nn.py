from scipy.special import expit
from scipy.stats import logistic
import random
import numpy as np
from pygraphviz import AGraph


sigmoid = expit
sigmoid_grad = logistic._pdf
random = random.Random()
random.seed(0)

class Node:
    name_counter = 0

    def __init__(self, function=sigmoid, gradient=sigmoid_grad):
        self.inputs = {}
        self.function = function
        self.gradient = gradient
        self.name = str(Node.name_counter)
        Node.name_counter += 1

    def add_input(self, port, weight):
        self.inputs[port] = weight

    def sum(self):
        return sum(key.poll() * weight for (key, weight) in self.inputs.items())

    def poll(self):
        return self.function(self.sum())

    def backprop(self, train_rate, base):
        grad = self.gradient(self.sum())
        updates = {}
        for input in self.inputs.keys():
            updates[input] = train_rate * base * grad * input.poll()
        for input in self.inputs.keys():
            input.backprop(train_rate, base * grad * self.inputs[input])
        for input in self.inputs.keys():
            self.inputs[input] -= updates[input]

class BiasNode(Node):
    def __init__(self, value=1):
        super().__init__()
        self.value = value

    def poll(self):
        return self.value

    def backprop(self, train_rate, base): pass

class InputNode(Node):
    def __init__(self, value=0):
        super().__init__()
        self.value = value

    def set(self, value):
        self.value = value

    def poll(self):
        return self.value

    def backprop(self, train_rate, base): pass

class OutputNode(Node):
    def __init__(self, target=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target

    def error(self):
        return .5 * (self.target - self.poll()) ** 2

    def error_gradient(self):
        return self.poll() - self.target

    def backprop(self, train_rate):
        updates = {}
        base = self.error_gradient() * self.gradient(self.sum())
        for input in self.inputs.keys():
            updates[input] = train_rate * base * input.poll()
        for input in self.inputs.keys():
            input.backprop(train_rate, base * self.inputs[input])
        for input in self.inputs.keys():
            self.inputs[input] -= updates[input]

class Layer:
    def __init__(self, num_nodes, node_type=Node, *args, **kwargs):
        self.nodes = [node_type(*args, **kwargs) for _ in range(num_nodes)]

    def connect_to_layer(self, layer, weight_func=random.random):
        for in_node in self.nodes:
            for out_node in layer.nodes:
                out_node.add_input(in_node, weight_func())

    def connect_to_node(self, node, weight_func=random.random):
        for in_node in self.nodes:
            node.add_input(in_node, weight_func())

class InputLayer(Layer):
    def __init__(self, num_nodes, values=None):
        super().__init__(num_nodes, node_type=InputNode)
        if values is not None:
            self.set(values)

    def set(self, values):
        assert(len(values) == len(self.nodes))
        for i, in_node in enumerate(self.nodes):
            in_node.set(values[i])

class OutputLayer(Layer):
    def __init__(self, num_nodes, values=None):
        super().__init__(num_nodes, node_type=OutputNode)
        if values is not None:
            self.set_target(values)

    def set_target(self, targets):
        assert(len(targets) == len(self.nodes))
        for i, out_node in enumerate(self.nodes):
            out_node.target = targets[i]

    #TODO: FIX for multiple output nodes
    def backprop(self, train_rate):
        self.nodes[0].backprop(train_rate)

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden_layers, hidden_layer_size, num_outputs, bias=True):
        self.input = InputLayer(num_inputs)
        self.hidden = [Layer(hidden_layer_size) for _ in range(num_hidden_layers)]
        self.output = OutputLayer(num_outputs)
        self.bias = BiasNode() if bias else None
        if num_hidden_layers == 0:
            self.input.connect_to_layer(self.output)
        else:
            self.input.connect_to_layer(self.hidden[0])
            for i in range(num_hidden_layers - 1):
                self.hidden[i].connect_to_layer(self.hidden[i+1])
            self.hidden[-1].connect_to_layer(self.output)
        if bias:
            for hidden_layer in self.hidden:
                for node in hidden_layer.nodes:
                    node.add_input(self.bias, random.random())
            for out_node in self.output.nodes:
                out_node.add_input(self.bias, random.random())

    def train(self, iterations, train_rate, input_samples, output_samples):
        num_samples = len(input_samples)
        assert(num_samples == len(output_samples))
        for step in range(iterations):
            print("\r{:.4}%".format(step/iterations*100), end="", flush=True)
            for i in range(num_samples):
                self.input.set(input_samples[i])
                self.output.set_target(output_samples[i])
                self.output.backprop(train_rate)
        print()

    def poll(self, input):
        self.input.set(input)
        return [out_node.poll() for out_node in self.output.nodes]

    def render(self, graph=None, start=None):
        if graph is None:
            graph = AGraph(directed=True)
        if start is None:
            for node in self.output.nodes:
                self.render(graph, node)
        else:
            if isinstance(start, InputNode):
                graph.add_node(start.name, label=str(start.poll()), color="green")
            elif isinstance(start, BiasNode):
                graph.add_node(start.name, label=str(start.poll()), color="orange")
            elif isinstance(start, OutputNode):
                graph.add_node(start.name, label="{:.3f}".format(float(start.poll())), color="red")
            else:
                graph.add_node(start.name, label="{:.3f}".format(float(start.poll())))
            for input in start.inputs.keys():
                self.render(graph, input)
                graph.add_edge(input.name, start.name, label="{:.3f}".format(start.inputs[input]))
        return graph


xor_net = NeuralNetwork(2, 1, 3, 1)
xor_net.train(50000, 0.05, [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]])
for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
   print(i, ":", xor_net.poll(i))
graph = xor_net.render()

graph.draw("xor.png", prog="dot")