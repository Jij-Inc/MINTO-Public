import abc


class Problem(abc.ABC):
    @abc.abstractmethod
    def problem(self):
        pass

    @abc.abstractmethod
    def random_data(self):
        pass
