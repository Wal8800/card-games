import abc

from bigtwo.bigtwo import BigTwoObservation


class BigTwoBot(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "action") and callable(subclass.action)

    @abc.abstractmethod
    def action(self, observation: BigTwoObservation):
        raise NotImplementedError("Action not implemented")
