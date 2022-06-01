from typing import Callable

import networkx as nx

from util.math_utils import Interval2D


class ImageSegmentationIterator:
    """
    Iterator on grid, implemented as interval tree.
    """

    __PROCESSED_CHILDREN = "processed_children"
    __PROCESSED = "processed"

    def __init__(self, interval: Interval2D, is_uniform: Callable[[Interval2D], bool]):

        self.interval_tree: nx.DiGraph = nx.DiGraph()

        self.root_node = interval

        self.interval_tree.add_node(self.root_node)

        self.current_node: Interval2D = self.root_node

        self.interval_tree.nodes[self.root_node][self.__PROCESSED_CHILDREN] = 0
        self.interval_tree.nodes[self.root_node][self.__PROCESSED] = False

        self.is_uniform = is_uniform

    def __iter__(self):
        return self

    @staticmethod
    def __is_empty_interval(interval: Interval2D) -> bool:
        return interval.max_x == interval.min_x and interval.min_y == interval.max_y

    def __climb_to_root(self):
        # Uniform node
        while self.current_node != self.root_node and self.interval_tree.nodes[self.current_node][self.__PROCESSED]:
            # We will have to climb upwards to the parent, and find the next non-free interval

            children = list(self.interval_tree.successors(self.current_node))

            if self.interval_tree.nodes[self.current_node][self.__PROCESSED_CHILDREN] >= len(children):
                # We always quadruple the interval
                self.interval_tree.nodes[self.current_node][self.__PROCESSED] = True
                self.current_node = next(self.interval_tree.predecessors(self.current_node))  # Go to parent
                self.interval_tree.nodes[self.current_node][self.__PROCESSED_CHILDREN] += 1

                children_parent = list(self.interval_tree.successors(self.current_node))

                if self.interval_tree.nodes[self.current_node][self.__PROCESSED_CHILDREN] >= len(children_parent):
                    self.interval_tree.nodes[self.current_node][self.__PROCESSED] = True
            else:
                break

    def __descend(self):

        children = list(self.interval_tree.successors(self.current_node))

        if self.interval_tree.nodes[self.current_node][self.__PROCESSED_CHILDREN] < len(children):
            self.current_node = children[self.interval_tree.nodes[self.current_node][self.__PROCESSED_CHILDREN]]

        if not self.interval_tree.nodes[self.current_node][self.__PROCESSED]:
            while not self.is_uniform(self.current_node) and not self.__is_empty_interval(self.current_node):

                mid_x = (self.current_node.min_x + self.current_node.max_x) // 2
                mid_y = (self.current_node.min_y + self.current_node.max_y) // 2

                new_nodes = list({
                    Interval2D(self.current_node.min_x, mid_x, self.current_node.min_y, mid_y),
                    Interval2D(self.current_node.min_x, mid_x,
                               min(mid_y + 1, self.current_node.max_y), self.current_node.max_y),
                    Interval2D(min(mid_x + 1, self.current_node.max_x), self.current_node.max_x,
                               self.current_node.min_y, mid_y),
                    Interval2D(min(mid_x + 1, self.current_node.max_x), self.current_node.max_x,
                               min(mid_y + 1, self.current_node.max_y), self.current_node.max_y),
                })

                for new_node in new_nodes:
                    self.interval_tree.add_edge(self.current_node, new_node)
                    self.interval_tree.nodes[new_node][self.__PROCESSED_CHILDREN] = 0
                    self.interval_tree.nodes[new_node][self.__PROCESSED] = False

                next_node = next(self.interval_tree.successors(self.current_node))

                self.current_node = next_node

    def __next__(self):

        self.__climb_to_root()

        self.__descend()

        children = list(self.interval_tree.successors(self.current_node))
        if self.current_node == self.root_node and \
                self.interval_tree.nodes[self.current_node][self.__PROCESSED_CHILDREN] >= len(children):

            raise StopIteration  # We have iterated the entire tree

        self.interval_tree.nodes[self.current_node][self.__PROCESSED] = True

        return self.current_node
