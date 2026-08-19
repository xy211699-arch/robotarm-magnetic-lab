from scripts.virtual_magnet.common import KEY_TO_ACTION, key_name_to_action


def test_required_keys_map_one_to_one():
    assert [KEY_TO_ACTION[str(index)] for index in range(10)] == list(range(10))
    assert KEY_TO_ACTION["-"] == 10
    assert len(set(KEY_TO_ACTION[key] for key in [str(i) for i in range(10)] + ["-"])) == 11
    assert key_name_to_action("KEY_7") == 7
    assert key_name_to_action("NUMPAD9") == 9
    assert key_name_to_action("MINUS") == 10
