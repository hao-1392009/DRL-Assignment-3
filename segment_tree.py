import abc


class SegmentTree(abc.ABC):
    def __init__(self, size, init_value):
        self.capacity = 1  # should be a power of 2 and >= size
        while self.capacity < size:
            self.capacity *= 2

        self.tree = [init_value for _ in range(2 * self.capacity)]

    def query(self, left=0, right=None):
        """
        Query the segment [left, right).
        If right is None, then query till the end.
        """
        if right is None:
            right = self.capacity
        right -= 1
        return self._query(left, right, 1, 0, self.capacity - 1)

    def __getitem__(self, index):
        return self.tree[self.capacity + index]

    def __setitem__(self, index, value):
        index += self.capacity
        self.tree[index] = value

        index //= 2
        while index >= 1:
            self.tree[index] = self._operation(self.tree[2 * index], self.tree[2 * index + 1])
            index //= 2

    @abc.abstractmethod
    def _operation(self, a, b):
        pass

    def _query(self, left, right, node, node_left, node_right):
        if left == node_left and right == node_right:
            return self.tree[node]

        node_mid = (node_left + node_right) // 2
        if right <= node_mid:
            return self._query(left, right, 2 * node, node_left, node_mid)
        elif node_mid + 1 <= left:
            return self._query(left, right, 2 * node + 1, node_mid + 1, node_right)
        else:
            return self._operation(
                self._query(left, node_mid, 2 * node, node_left, node_mid),
                self._query(node_mid + 1, right, 2 * node + 1, node_mid + 1, node_right)
            )

class SumSegmentTree(SegmentTree):
    def __init__(self, size):
        super().__init__(size, 0)
        self.size = size

    def min_index_greater_than(self, greater_than):
        index = 1
        while index < self.capacity:  # not a leaf node
            left = 2 * index
            right = left + 1

            if self.tree[left] > greater_than:
                index = left
            else:
                greater_than -= self.tree[left]
                index = right

        # If greater_than > self.query() (e.g. due to precision error), we would try to
        # return the right-most node, so we have to clip it.
        return min(index - self.capacity, self.size - 1)

    def _operation(self, a, b):
        return a + b

class MinSegmentTree(SegmentTree):
    def __init__(self, size):
        super().__init__(size, float("inf"))

    def _operation(self, a, b):
        return min(a, b)
