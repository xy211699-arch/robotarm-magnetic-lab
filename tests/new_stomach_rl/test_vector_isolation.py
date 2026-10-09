from types import SimpleNamespace
import numpy as np
import pytest
import torch
from robotarm_magnetic_lab.runtime.new_stomach_rl_vector_rows import (
    RowAsset, RowData, row_ids, translate_pose)


def test_row_reads_and_writes_cannot_escape_or_alias_another_row():
    calls = []
    asset = SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.arange(18).reshape(2,9)),
        joint_names=['j1']), permanent_wrench_composer=SimpleNamespace(reset=lambda **kw: calls.append(kw)),
        set_joint_position_target_index=lambda **kw: calls.append(kw))
    row = RowAsset(asset, 1, 2, 'cpu')
    assert row.data.joint_pos.torch.shape == (1,9)
    assert row.data.joint_pos.torch[0,0] == 9
    row.set_joint_position_target_index(target=torch.zeros(1,9),joint_ids=list(range(9)))
    assert calls[-1]['env_ids'].tolist() == [1]
    row.permanent_wrench_composer.reset()
    assert calls[-1]['env_ids'].tolist() == [1]
    with pytest.raises(ValueError):
        row.set_joint_position_target_index(target=torch.zeros(1,9), env_ids=[1])
    with pytest.raises(AttributeError):
        row.write_root_pose_to_sim_index


def test_pose_translation_roundtrip_and_row_permutation():
    pose = np.array([1.,.2,.05,0,0,0,1.])
    origins = [[-2.,0,0],[2.,0,0]]
    translated = [translate_pose(pose, [0,0,0], origin) for origin in origins]
    for value, origin in zip(translated, origins):
        assert np.allclose(translate_pose(value, origin, [0,0,0]), pose)
        assert np.array_equal(value[3:], pose[3:])
    assert np.array_equal(translated[::-1], [translate_pose(pose,[0,0,0],o) for o in origins[::-1]])


def test_bad_row_and_ambiguous_tensor_are_rejected():
    for ids in ([0,0],[-1],[2],[.5]):
        with pytest.raises(ValueError):
            row_ids(ids,2)
    with pytest.raises(ValueError):
        RowData(SimpleNamespace(value=torch.zeros(3,9)),0,2).value
