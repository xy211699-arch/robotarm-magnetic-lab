"""Read-only P1 host attribution; CUDA completion is measured at outer boundaries.

Nested host durations are not GPU kernel durations and must never be summed
as disjoint phase costs. No synchronization is inserted between physical steps.
"""
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import time


class CapacityProfile:
    def __init__(self, enabled=False, clock=time.perf_counter, synchronize=lambda: None,
                 annotate=lambda label: nullcontext()):
        self.enabled, self.clock, self.synchronize = bool(enabled), clock, synchronize
        self.records = []
        self.current = None
        self.stack = []
        self.annotate = annotate

    @contextmanager
    def boundary(self, valid_rows, phase, trace=False):
        if phase not in ('active', 'warmup', 'reset') or int(valid_rows) != valid_rows or valid_rows < 0:
            raise ValueError('finite nonnegative valid row count and known phase required')
        if phase != 'active' and valid_rows:
            raise ValueError('reset/warmup cannot produce policy samples')
        if self.current is not None:
            raise RuntimeError('overlapping profile boundaries are forbidden')
        if self.enabled:
            self.synchronize()
        started = self.clock() if self.enabled else None
        row = dict(phase=phase, valid_rows=int(valid_rows), trace=bool(trace), completed=False,
                   calls={}, timers={}, wall_s=None, unattributed_outer_s=None)
        self.current = row
        try:
            yield row
            row['completed'] = True
        finally:
            try:
                if self.enabled:
                    self.synchronize()
                    row['wall_s'] = self.clock() - started
                    if row['wall_s'] < 0:
                        raise RuntimeError('nonmonotonic timing clock')
                    exclusive = sum(x['exclusive_host_s'] for x in row['timers'].values())
                    # Includes CUDA completion drain, not purely CPU work.
                    row['unattributed_outer_s'] = max(row['wall_s'] - exclusive, 0.)
            except BaseException:
                row['completed'] = False
                raise
            finally:
                self.records.append(row)
                self.current = None
                self.stack.clear()

    @contextmanager
    def scope(self, label):
        if self.current is None:
            yield
            return
        row = self.current
        row['calls'][label] = row['calls'].get(label, 0) + 1
        if not self.enabled:
            yield
            return
        parent = self.stack[-1]['label'] if self.stack else 'outer_boundary'
        frame = dict(label=label, started=self.clock(), child_s=0.)
        self.stack.append(frame)
        try:
            with self.annotate(label) if row['trace'] else nullcontext():
                yield
        finally:
            elapsed = self.clock() - frame['started']
            self.stack.pop()
            if self.stack:
                self.stack[-1]['child_s'] += elapsed
            value = row['timers'].setdefault(label, dict(inclusive_s=0., exclusive_host_s=0., parents={}))
            value['inclusive_s'] += elapsed
            value['exclusive_host_s'] += max(elapsed-frame['child_s'], 0.)
            value['parents'][parent] = value['parents'].get(parent, 0) + 1

    @contextmanager
    def patch(self, obj, attribute, label):
        original = getattr(obj, attribute)
        def wrapped(*args, **kwargs):
            with self.scope(label):
                return original(*args, **kwargs)
        setattr(obj, attribute, wrapped)
        try:
            yield
        finally:
            setattr(obj, attribute, original)

    def summary(self):
        good = [r for r in self.records if r['completed']]
        steady = [r for r in good if r['phase'] == 'active' and not r['trace']]
        valid = sum(r['valid_rows'] for r in good)
        count = sum(r['valid_rows'] for r in steady)
        wall = (sum(r['wall_s'] for r in self.records) if self.enabled and
                all(r['wall_s'] is not None for r in self.records) else None)
        steady_wall = sum(r['wall_s'] for r in steady) if self.enabled else None
        return dict(timing_enabled=self.enabled, valid_transitions=valid, accounted_wall_s=wall,
                    q_valid_including_reset=valid/wall if wall else None,
                    steady_transitions=count, steady_wall_s=steady_wall,
                    steady_transitions_per_s=count/steady_wall if steady_wall else None,
                    trace_boundaries=sum(r['trace'] for r in self.records),
                    failed_boundaries=sum(not r['completed'] for r in self.records),
                    scope='diagnostic env.step/reset only; excludes app startup and file persistence',
                    attribution='inclusive/exclusive host spans; nested spans not additive; CUDA completed only outside')


def validate_worker_summary(path, process_exit):
    path = Path(path)
    if not path.is_file():
        raise ValueError('worker missing persistent summary')
    if process_exit != 0:
        raise ValueError(f'worker process exit {process_exit}, not success')
    def invalid(value):
        raise ValueError(f'nonfinite worker summary: {value}')
    data = json.loads(path.read_text(), parse_constant=invalid)
    if data.get('status') != 'pass':
        raise ValueError(f'worker status is not pass: {data.get("status")}')
    return data
