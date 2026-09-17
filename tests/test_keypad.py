from festival_pet.keypad import DEFAULT_KEYMAP, HOLD_S, KeyEvent, KeyMap, keyboard_devices


def ev(key, down, t):
    return KeyEvent(key, down, t, "test")


def test_press_only_keys_fire_on_key_down():
    km = KeyMap()
    assert km.feed(ev("B", True, 10.0), 10.0) == ("tap", 10.0)  # no release latency for beats
    assert km.feed(ev("B", False, 10.1), 10.1) is None
    assert km.feed(ev("SPACE", True, 11.0), 11.0) == ("tap", 11.0)


def test_press_and_hold_keys():
    km = KeyMap()
    assert km.feed(ev("A", True, 10.0), 10.0) is None  # A has a hold action: wait for release
    assert km.feed(ev("A", False, 10.2), 10.2) == ("tilt_left", 10.0)
    assert km.feed(ev("A", True, 20.0), 20.0) is None
    assert km.tick(20.5) == []
    assert km.tick(20.0 + HOLD_S + 0.01) == [("manual_groove", 20.0)]  # fires at the hold time, not on release
    assert km.tick(21.5) == []  # once
    assert km.feed(ev("A", False, 21.6), 21.6) is None  # the release after a hold does nothing


def test_unmapped_keys_are_ignored_and_map_edits_validate():
    km = KeyMap()
    assert km.feed(ev("Q", True, 1.0), 1.0) is None
    km.set("q", "happy", "none")
    assert km.feed(ev("Q", True, 2.0), 2.0) == ("happy", 2.0)
    km.set("Q", "none", "none")
    assert "Q" not in km.keys
    import pytest
    with pytest.raises(KeyError):
        km.set("A", "dance", "none")
    with pytest.raises(KeyError):
        km.set("BANANA", "tap", "none")
    assert set(DEFAULT_KEYMAP) >= {"A", "B", "C", "D"}


def test_keyboard_discovery_does_not_crash_off_robot():
    assert isinstance(keyboard_devices(), list)
