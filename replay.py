"""prioritized experience replay buffer, use tensorflow API"""


from collections import namedtuple
import numpy as np


Transition = namedtuple("Transition", ["s", "a", "r", "s_", "done"])


class UniformReplayBuffer(object):

    """uniform replay buffer.
    
    Buffer is circular.

    Sample all the stored experiences with equal probability.
    """

    def __init__(self, size: int):
        self.size = size
        self.buffer = [None for _ in range(self.size)]
        self.index = 0
        self.full = False
        self.per = False

    def append(self, transition):
        """append a transition (namedtuple)"""
        self.buffer[self.index] = transition
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0

    def sample(self, batch_size: int):
        """sample experiences.

        return Dict[str: np.ndarray].
        keys in ['s', 'a', 'r', 's_', 'done']
        """
        capacity = self.size if self.full else self.index
        indices = np.random.choice(range(capacity), size=batch_size)
        return self._reformat(indices), indices, None

    def _reformat(self, indices):
        """format the output"""
        return {
            field_name: np.array([getattr(self.buffer[i], field_name) for i in indices])
                for field_name in Transition._fields
        }

    @property
    def nb_frames(self):
        """get the number of stored experiences"""
        return self.size if self.full else self.index


class PrioritizedReplayBuffer(object):

    """ prioritized replay buffer (PER).

    Buffer is circular, but the priority is segment tree based
    """

    def __init__(self, size: int, beta: float = 0.4, alpha: float = 1.0, eps: float = 0.01):
        self.size = size
        self.index = 0
        self.full = False
        self.buffer = [None for _ in range(self.size)] 
        self.base_node, self.leaf_nodes = create_tree([0 for i in range(self.size)]) # self.leaf_nodes.idx refers to self.buffer[idx]
        self.beta = beta
        self.alpha = alpha
        self.eps = eps
        self.per = True

    def append(self, experience: Transition, priority: float):
        self.buffer[self.index] = experience
        self.update(self.index, priority)
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0

    def update(self, idx: int, priority: float):
        """update segment tree"""
        update(self.leaf_nodes[idx], self._adjust_priority(priority))

    def _adjust_priority(self, p):
        return (np.fabs(p) + self.eps) ** self.alpha

    def sample(self, batch_size: int):
        sampled_idxs = []
        priorities = []
        segment = self.base_node.value / batch_size
        for i in range(batch_size):
            # sample randomly
            s = np.random.uniform(i * segment, (i+1) * segment)
            # get the sample node from the segment tree
            sample_node = retrieve(s, self.base_node)
            # store idx and priority
            sampled_idxs.append(sample_node.idx)
            priorities.append(sample_node.value / self.base_node.value)
        capacity = self.size if self.full else self.index
        is_weights = np.power(capacity * np.array(priorities), -self.beta)
        is_weights /= is_weights.max()
        # return Tuple[Dict[str: np.ndarray], List[int], np.ndarray]
        return self._reformat(sampled_idxs), sampled_idxs, is_weights

    @property
    def nb_frames(self):
        return self.size if self.full else self.index
        
    def _reformat(self, sampled_idxs):
        return {
            field_name: np.array([getattr(self.buffer[i], field_name) for i in sampled_idxs])
                for field_name in Transition._fields
        }


class Node:

    """node for segment tree"""
    
    def __init__(self, left, right, is_leaf: bool = False, idx = None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.value = sum(n.value for n in (left, right) if n is not None)
        self.parent = None
        self.idx = idx  # this value is only set for leaf nodes
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self

    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value
        return leaf


def create_tree(input: list):
    nodes = [Node.create_leaf(v, i) for i, v in enumerate(input)]
    leaf_nodes = nodes
    while len(nodes) > 1:
        lefts = nodes[::2]
        rights = nodes[1::2]
        halfsize = min(len(lefts), len(rights))
        upper_layer = []
        for _ in range(halfsize):
            upper_layer.append(Node(lefts.pop(0), rights.pop(0)))
        if len(lefts) > 0:
            upper_layer.append(lefts.pop(0))
        nodes = upper_layer
    return nodes[0], leaf_nodes


def retrieve(value: float, node: Node):
    if node.is_leaf:
        return node
    
    if node.left.value >= value: 
        return retrieve(value, node.left)
    else:
        return retrieve(value - node.left.value, node.right)


def update(node: Node, new_value: float):
    change = new_value - node.value

    node.value = new_value
    propagate_changes(change, node.parent)


def propagate_changes(change: float, node: Node):
    node.value += change

    if node.parent is not None:
        propagate_changes(change, node.parent)