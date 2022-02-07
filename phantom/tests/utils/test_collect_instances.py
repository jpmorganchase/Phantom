from dataclasses import dataclass

from phantom.utils import (
    collect_instances_of_type,
    collect_instances_of_type_with_paths,
)


@dataclass
class TestDataclass:
    x: str


def test_collect_instances_of_type():
    # Object
    x = "a"
    assert collect_instances_of_type(str, x) == ["a"]

    # Dict
    x = {"x": "a", "y": "b"}
    assert collect_instances_of_type(str, x) == ["a", "b"]

    # List
    x = ["a", "b"]
    assert collect_instances_of_type(str, x) == ["a", "b"]

    # Tuple
    x = ("a", "b")
    assert collect_instances_of_type(str, x) == ["a", "b"]

    # Dataclass
    x = TestDataclass("a")
    assert collect_instances_of_type(str, x) == ["a"]

    # Nested Object
    x = [{"x": "a"}, {"y": "b"}]
    assert collect_instances_of_type(str, x) == ["a", "b"]

    # Duplicate object
    x = [123, 123]
    print(collect_instances_of_type(int, x))
    assert collect_instances_of_type(int, x) == [123]


def test_collect_instances_of_type_with_paths():
    # Object
    x = "a"
    assert collect_instances_of_type_with_paths(str, x) == [("a", [[]])]

    # Dict
    x = {"x": "a", "y": "b"}
    assert collect_instances_of_type_with_paths(str, x) == [
        ("a", [[(True, "x")]]),
        ("b", [[(True, "y")]]),
    ]

    # List
    x = ["a", "b"]
    assert collect_instances_of_type_with_paths(str, x) == [("a", [[0]]), ("b", [[1]])]

    # Tuple
    x = ("a", "b")
    assert collect_instances_of_type_with_paths(str, x) == [("a", [[0]]), ("b", [[1]])]

    # Dataclass
    x = TestDataclass("a")
    assert collect_instances_of_type_with_paths(str, x) == [("a", [[(False, "x")]])]

    # Nested Object
    x = [{"x": "a"}, {"y": "b"}]
    assert collect_instances_of_type_with_paths(str, x) == [
        ("a", [[0, (True, "x")]]),
        ("b", [[1, (True, "y")]]),
    ]

    # Duplicate object
    x = [123, 123]
    assert collect_instances_of_type_with_paths(int, x) == [(123, [[0], [1]])]
