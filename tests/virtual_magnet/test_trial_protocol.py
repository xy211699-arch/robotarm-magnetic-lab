from scripts.virtual_magnet.common import generate_manifest, manifest_digest


def test_manifests_are_deterministic_and_split_disjoint():
    development = generate_manifest("development", 20, base_seed=7000)
    held_out = generate_manifest("held-out", 20, base_seed=7000)
    assert len(development) == 220
    assert len(held_out) == 220
    assert manifest_digest(development) == manifest_digest(
        generate_manifest("development", 20, base_seed=7000)
    )
    assert {item.seed for item in development}.isdisjoint(item.seed for item in held_out)


def test_invalid_move_classes_are_separate():
    positive = generate_manifest("held-out-invalid-pos", 20, base_seed=9000, action_ids=(9,), valid_move=False)
    negative = generate_manifest("held-out-invalid-neg", 20, base_seed=9000, action_ids=(10,), valid_move=False)
    assert len(positive) == len(negative) == 20
    assert {item.seed for item in positive}.isdisjoint(item.seed for item in negative)

