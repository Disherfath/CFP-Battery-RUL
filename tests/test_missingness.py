import numpy as np
import pytest

from batt_rul.missingness import (
    apply_history_missingness,
    apply_missingness,
    delta_t_from_mask,
    interpolate_history,
)


@pytest.fixture
def curve():
    return np.arange(400, dtype=float).reshape(100, 4)


@pytest.mark.parametrize("kind", ["mcar", "mcar_time", "block", "window"])
def test_requested_rate_is_respected(curve, kind):
    result = apply_missingness(curve, kind, 0.2, np.random.default_rng(7))
    assert abs(result.realized_rate - 0.2) <= 0.005
    np.testing.assert_array_equal(result.values[result.observed], curve[result.observed])
    assert np.isnan(result.values[~result.observed]).all()


def test_time_mcar_is_random_in_time_but_synchronous_across_channels(curve):
    result = apply_missingness(curve, "mcar_time", 0.2, np.random.default_rng(7))
    assert np.all(result.observed == result.observed[:, [0]])
    missing_rows = np.flatnonzero(~result.observed[:, 0])
    assert len(missing_rows) == 20
    assert np.any(np.diff(missing_rows) > 1)


def test_block_is_one_contiguous_interval_across_channels(curve):
    result = apply_missingness(curve, "block", 0.2, np.random.default_rng(3))
    missing_rows = np.flatnonzero(~result.observed[:, 0])
    np.testing.assert_array_equal(np.diff(missing_rows), np.ones(len(missing_rows) - 1))
    assert np.all(result.observed == result.observed[:, [0]])


def test_channel_dropout_removes_only_requested_channel(curve):
    result = apply_missingness(curve, "channel", 0.2, np.random.default_rng(0), channel=2)
    assert (~result.observed[:, 2]).all()
    assert result.observed[:, [0, 1, 3]].all()


def test_delta_t_accumulates_and_resets():
    observed = np.array([[True], [False], [False], [True], [False]])
    coordinates = np.array([0.0, 0.5, 1.5, 2.0, 4.5])
    gaps = delta_t_from_mask(observed, coordinates)
    np.testing.assert_allclose(gaps[:, 0], [0.0, 0.5, 1.5, 0.0, 2.5])


def test_invalid_rate_fails(curve):
    with pytest.raises(ValueError):
        apply_missingness(curve, "mcar", 1.0, np.random.default_rng(0))


def test_history_corruption_is_deterministic_and_keeps_time():
    curves = np.ones((5, 100, 4), dtype=float)
    curves[:, :, 3] = np.linspace(0.0, 10.0, 100)
    first = apply_history_missingness(
        curves, "block", 0.2, base_seed=17, cell_id="cell-a", replicate=2
    )
    second = apply_history_missingness(
        curves, "block", 0.2, base_seed=17, cell_id="cell-a", replicate=2
    )
    np.testing.assert_array_equal(first.observed, second.observed)
    assert np.isclose(first.realized_sensor_rate, 0.2)
    assert first.observed[:, :, 3].all()
    np.testing.assert_array_equal(first.values[:, :, 3], curves[:, :, 3])


def test_channel_dropout_is_consistent_across_history_and_interpolates():
    curves = np.ones((3, 32, 4), dtype=float)
    curves[:, :, 3] = np.linspace(0.0, 1.0, 32)
    corrupted = apply_history_missingness(
        curves,
        "channel",
        1.0 / 3.0,
        base_seed=4,
        cell_id="cell-b",
        replicate=0,
        dropped_channel=1,
    )
    assert (~corrupted.observed[:, :, 1]).all()
    assert corrupted.observed[:, :, [0, 2, 3]].all()
    fill_values = np.array([10.0, 20.0, 30.0, 40.0])
    filled = interpolate_history(corrupted, fully_missing_fill=fill_values)
    assert np.isfinite(filled).all()
    assert (filled[:, :, 1] == 20.0).all()


def test_history_channel_dropout_requires_explicit_channel_and_realized_rate():
    curves = np.ones((3, 32, 4), dtype=float)
    curves[:, :, 3] = np.linspace(0.0, 1.0, 32)
    with pytest.raises(ValueError, match="explicit dropped_channel"):
        apply_history_missingness(
            curves,
            "channel",
            1.0 / 3.0,
            base_seed=4,
            cell_id="cell-b",
            replicate=0,
        )
    with pytest.raises(ValueError, match="realized rate"):
        apply_history_missingness(
            curves,
            "channel",
            0.2,
            base_seed=4,
            cell_id="cell-b",
            replicate=0,
            dropped_channel=1,
        )
