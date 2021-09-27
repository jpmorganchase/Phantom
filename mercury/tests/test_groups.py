import pytest
import typing as _t

from mercury import Path, Groups


def test_flat():
    gps = Groups()

    # Adding:
    assert not gps.is_group("G1")
    assert not gps.is_group("G2")

    gps.add("G1", ["Actor1", "Actor2"])

    assert gps.is_group("G1")
    assert not gps.is_group("G2")

    assert gps["G1"] == {"Actor1", "Actor2"}

    gps.add("G2", ["Actor2", "Actor3"])

    assert gps.is_group("G1")
    assert gps.is_group("G2")

    assert gps["G1"] == {"Actor1", "Actor2"}
    assert gps["G2"] == {"Actor2", "Actor3"}

    # Adding (error):
    with pytest.raises(KeyError):
        gps.add("G1", ["Actor1"])

    with pytest.raises(KeyError):
        gps.add("G2", ["Actor3"])

    # Removing:
    gps.remove("G1")

    assert not gps.is_group("G1")
    assert gps.is_group("G2")

    assert gps["G2"] == {"Actor2", "Actor3"}

    gps.remove("G2")

    assert not gps.is_group("G1")
    assert not gps.is_group("G2")

    # Removing (error):
    with pytest.raises(KeyError):
        gps.remove("G1")

    with pytest.raises(KeyError):
        gps.remove("G2")


def test_recursive():
    gps = Groups()

    # Adding:
    gps.add("Foo", {"A"})

    assert gps["Foo"] == {"A"}

    gps.add("Foo::Bar", {"A", "B", "C", "D"})
    gps.add("Foo::Bar::Baz", {"A", "C"})

    assert gps["Foo"] == {"A", "B", "C", "D"}
    assert gps["Foo::Bar"] == {"A", "B", "C", "D"}
    assert gps["Foo::Bar::Baz"] == {"A", "C"}

    gps.add("Foo::Bar::Boo", {"E"})

    assert gps["Foo"] == {"A", "B", "C", "D", "E"}
    assert gps["Foo::Bar"] == {"A", "B", "C", "D", "E"}
    assert gps["Foo::Bar::Baz"] == {"A", "C"}
    assert gps["Foo::Bar::Boo"] == {"E"}

    # Adding (error):
    with pytest.raises(KeyError):
        gps.add("Foo::Bar::Baz", ["A"])

    with pytest.raises(KeyError):
        gps.add("Foo", ["A"])

    # Removing:
    gps.remove("Foo::Bar")

    assert gps.is_group("Foo")
    assert not gps.is_group("Foo::Bar")
    assert not gps.is_group("Foo::Bar::Baz")
    assert not gps.is_group("Foo::Bar::Boo")

    assert len(gps["Foo"]) == 0

    # Association:
    gps.assign("Foo", {"A"})

    assert gps["Foo"] == {"A"}


def test_deep_assoc():
    gps = Groups()

    gps.add("Foo::Bar::Baz", {"A", "C"})
    gps.add("Foo::Bar::Boo", {"E"})

    gps.assign("Foo::Bar", {"Z"})

    assert gps["Foo::Bar"] == {"A", "C", "E", "Z"}
    assert gps["Foo::Bar::Baz"] == {"A", "C"}
    assert gps["Foo::Bar::Boo"] == {"E"}

    gps.assign("Foo::Bar::Boo", {"Y"})

    assert gps["Foo::Bar"] == {"A", "C", "E", "Z"}
    assert gps["Foo::Bar::Baz"] == {"A", "C"}
    assert gps["Foo::Bar::Boo"] == {"E", "Y"}
