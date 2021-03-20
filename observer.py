class Observer(object):
    def __init__(self, size=10):
        self.size = size
        self.buffer = [0 for _ in range(size)]
        self.index = 0
        self.full = False
    
    def append(self, reward):
        self.buffer[self.index] = reward
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0

    @property
    def result(self):
        raise NotImplementedError


class AverageObserver(Observer):
    def __init__(self, size=10):
        super(AverageObserver, self).__init__(size)
        self.name = 'average'

    @property
    def result(self):
        nb = self.size if self.full else self.index
        return sum(self.buffer) / nb


class MaximumObserver(Observer):
    def __init__(self, size=10):
        super(MaximumObserver, self).__init__(size)
        self.name = "maximum"

    @property
    def result(self):
        return max(self.buffer)


class HistoryObserver(Observer):
    def __init__(self, size=10):
        super(HistoryObserver, self).__init__(size)
        self.name = "history"
    
    @property
    def result(self):
        return self.buffer