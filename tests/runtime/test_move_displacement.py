import numpy as np
import pytest

from robotarm_magnetic_lab.runtime.move_displacement import corrected_move_displacement


def test_corrected_move_displacement_subtracts_paired_hold_drift():
    result = corrected_move_displacement(
        active_start_com=[1.0, 2.0, 3.0],
        active_end_com=[1.004, 2.001, 3.0],
        hold_start_com=[1.0, 2.0, 3.0],
        hold_end_com=[1.0005, 2.0, 3.0],
        command_direction_world=[2.0, 0.0, 0.0],
    )
    assert result.active_signed_m == pytest.approx(0.004)
    assert result.hold_signed_m == pytest.approx(0.0005)
    assert result.corrected_signed_m == pytest.approx(0.0035)


def test_corrected_move_displacement_ignores_transverse_motion():
    result = corrected_move_displacement(
        active_start_com=np.zeros(3),
        active_end_com=[0.0, -0.002, 0.004],
        hold_start_com=np.zeros(3),
        hold_end_com=[0.0, -0.0002, -0.003],
        command_direction_world=[0.0, -1.0, 0.0],
    )
    assert result.corrected_signed_m == pytest.approx(0.0018)


@pytest.mark.parametrize("direction", ([0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]))
def test_corrected_move_displacement_rejects_invalid_direction(direction):
    with pytest.raises(ValueError):
        corrected_move_displacement(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), direction)
