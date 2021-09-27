from abc import abstractmethod, ABC
from typing import TypeVar


class Type(ABC):
    """
    Abstract base type class.
    """

    pass


T = TypeVar("T", bound=Type)


class Supertype(ABC):
    @abstractmethod
    def sample(self) -> T:
        """
        Base method for sampling a Type from a Supertype.

        Must be implemented by supertypes that inherit from this class.
        """
        raise NotImplementedError


class NullType(Type):
    """
    An implementation of Type that holds no values.
    """

    pass


class NullSupertype(Supertype):
    """
    An implementation of Supertype that returns a type that holds no values.
    """

    def sample(self) -> NullType:
        """
        Returns a NullType that holds no values.
        """
        return NullType()
