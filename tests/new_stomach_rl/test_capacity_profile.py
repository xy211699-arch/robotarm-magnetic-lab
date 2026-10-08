"""P1 CPU timing/accounting tests; never launch Kit or a GPU worker."""
import json
from contextlib import nullcontext

import pytest

from robotarm_magnetic_lab.runtime.new_stomach_rl_capacity_profile import (
    CapacityProfile, validate_worker_summary,
)


def test_nested_inclusive_times_are_not_added_and_cuda_sync_is_outer_only():
    time = [0.]
    sync = []
    p = CapacityProfile(enabled=True, clock=lambda: time[0], synchronize=lambda: sync.append(time[0]))
    with p.boundary(valid_rows=1, phase='active'):
        with p.scope('magnetic'):
            time[0] += 1
            with p.scope('finite_model'):
                time[0] += 2
            time[0] += 1
        time[0] += 3
    row = p.records[0]
    assert sync == [0., 7.]
    assert row['wall_s'] == 7
    assert row['timers']['magnetic']['inclusive_s'] == 4
    assert row['timers']['magnetic']['exclusive_host_s'] == 2
    assert row['timers']['finite_model']['inclusive_s'] == 2
    assert row['timers']['finite_model']['parents'] == {'magnetic': 1}
    assert row['unattributed_outer_s'] == 3


def test_reset_padding_and_profile_trace_are_not_steady_state_samples():
    time = [0.]
    p = CapacityProfile(enabled=True, clock=lambda: time[0])
    for phase, valid, trace, duration in [('reset', 0, False, 2), ('active', 2, False, 3), ('active', 2, True, 5)]:
        with p.boundary(valid_rows=valid, phase=phase, trace=trace):
            time[0] += duration
    s = p.summary()
    assert s['valid_transitions'] == 4
    assert s['accounted_wall_s'] == 10
    assert s['q_valid_including_reset'] == .4
    assert s['steady_transitions'] == 2
    assert s['steady_wall_s'] == 3
    assert s['trace_boundaries'] == 1


def test_timing_disabled_calls_original_without_cuda_sync_or_clock():
    def forbidden():
        raise AssertionError('disabled timing must not synchronize or read clock')
    p = CapacityProfile(enabled=False, clock=forbidden, synchronize=forbidden)
    class Object:
        def run(self, value):
            return value + 1
    obj = Object()
    original = obj.run
    with p.patch(obj, 'run', 'simulation'):
        with p.boundary(valid_rows=1, phase='active'):
            assert obj.run(2) == 3
    assert obj.run == original
    assert p.records[0]['calls'] == {'simulation': 1}
    assert p.records[0]['wall_s'] is None
    assert p.summary()['q_valid_including_reset'] is None


def test_exception_restores_original_and_does_not_publish_a_successful_boundary():
    p = CapacityProfile(enabled=True)
    class Object:
        def run(self):
            raise ValueError('worker fault')
    obj = Object()
    original = obj.run
    with pytest.raises(ValueError, match='worker fault'):
        with p.patch(obj, 'run', 'simulation'):
            with p.boundary(valid_rows=1, phase='active'):
                obj.run()
    assert obj.run == original
    assert p.records[0]['completed'] is False
    assert p.summary()['valid_transitions'] == 0


def test_worker_missing_summary_or_bad_exit_is_never_pass(tmp_path):
    file = tmp_path / 'summary.json'
    with pytest.raises(ValueError, match='missing'):
        validate_worker_summary(file, 0)
    file.write_text(json.dumps({'status': 'pass'}))
    with pytest.raises(ValueError, match='exit'):
        validate_worker_summary(file, 1)
    assert validate_worker_summary(file, 0)['status'] == 'pass'
    file.write_text(json.dumps({'status': 'interrupted'}))
    with pytest.raises(ValueError, match='status'):
        validate_worker_summary(file, 0)


def test_boundary_arguments_and_reentrant_measurement_are_rejected():
    p = CapacityProfile(enabled=False)
    with pytest.raises(ValueError):
        with p.boundary(valid_rows=1, phase='reset'):
            pass
    with p.boundary(valid_rows=1, phase='active'):
        with pytest.raises(RuntimeError):
            with p.boundary(valid_rows=1, phase='active'):
                pass


def test_profiler_annotations_only_on_explicit_trace_boundary():
    labels = []
    def annotate(label):
        labels.append(label)
        return nullcontext()
    p = CapacityProfile(enabled=True, annotate=annotate)
    for traced in (False, True):
        with p.boundary(valid_rows=1, phase='active', trace=traced):
            with p.scope('magnetic'):
                pass
    assert labels == ['magnetic']


def test_cuda_completion_failure_does_not_count_a_transition_or_publish_throughput():
    syncs = []
    def synchronize():
        syncs.append(1)
        if len(syncs) == 2:
            raise RuntimeError('fake CUDA completion fault')
    p = CapacityProfile(enabled=True, synchronize=synchronize)
    with pytest.raises(RuntimeError, match='completion fault'):
        with p.boundary(valid_rows=1, phase='active'):
            pass
    assert p.records[0]['completed'] is False
    assert p.summary()['valid_transitions'] == 0
    assert p.summary()['q_valid_including_reset'] is None
