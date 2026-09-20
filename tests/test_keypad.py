import pytest

from festival_pet.keypad import ACTIONS, DEFAULT_LAYER_KEYS, LAYERS, KeyEvent, KeyMap, keyboard_devices


def ev(key, down, t):
    return KeyEvent(key, down, t, "test")


def test_three_layers_of_four_fixed_actions():
    assert list(LAYERS) == ["dancing", "petting", "caring"] and all(len(a) == 4 for a in LAYERS.values())  # the pad's order: dancing, then petting
    assert len(set(ACTIONS)) == 12
    assert set(DEFAULT_LAYER_KEYS) == set(LAYERS)


def test_keys_fire_on_key_down_only_and_release_does_nothing():
    km = KeyMap()
    assert km.feed(ev("B", True, 10.0), 10.0) == ("tap", 10.0)  # no release latency for beats
    assert km.feed(ev("B", False, 10.1), 10.1) is None
    assert km.feed(ev("A", True, 11.0), 11.0) == ("groove_left", 11.0)
    assert km.feed(ev("A", False, 11.9), 11.9) is None  # no hold actions any more
    assert km.feed(ev("I", True, 12.0), 12.0) == ("snack", 12.0)
    assert km.feed(ev("H", True, 13.0), 13.0) == ("belly_rub", 13.0)
    assert km.feed(ev("Q", True, 14.0), 14.0) is None  # not on the pad
    assert km.lookup("J") == ("caring", "mushroom") and km.lookup("Q") is None


def test_setting_a_slot_moves_the_key_and_validates():
    km = KeyMap()
    km.set("caring", 0, "f1")
    assert km.layers["caring"] == ["F1", "J", "K", "L"]
    km.set("petting", 3, "A")  # A was the dancing layer's first key: taken away from there
    assert km.layers["dancing"][0] == "" and km.layers["petting"][3] == "A"
    assert km.feed(ev("A", True, 1.0), 1.0) == ("belly_rub", 1.0)
    with pytest.raises(KeyError):
        km.set("sleeping", 0, "A")
    with pytest.raises(KeyError):
        km.set("caring", 4, "A")
    with pytest.raises(KeyError):
        km.set("caring", 0, "BANANA")


def test_keyboard_discovery_does_not_crash_off_robot():
    assert isinstance(keyboard_devices(), list)
