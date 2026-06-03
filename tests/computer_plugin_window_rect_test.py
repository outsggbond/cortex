from __future__ import annotations

from system.computer_use import plugins


def test_window_rect_prefers_dwm_bounds() -> None:
    original_dwm = plugins._dwm_window_rect_windows
    original_raw = plugins._raw_window_rect_windows
    try:
        plugins._dwm_window_rect_windows = lambda hwnd: (766, 100, 1406, 1268)
        plugins._raw_window_rect_windows = lambda hwnd: (_ for _ in ()).throw(
            AssertionError("raw rect should not be used when DWM bounds are available")
        )

        assert plugins._window_rect_windows(132628) == (766, 100, 1406, 1268)
    finally:
        plugins._dwm_window_rect_windows = original_dwm
        plugins._raw_window_rect_windows = original_raw


def test_window_rect_falls_back_to_raw_bounds() -> None:
    original_dwm = plugins._dwm_window_rect_windows
    original_raw = plugins._raw_window_rect_windows
    try:
        plugins._dwm_window_rect_windows = lambda hwnd: None
        plugins._raw_window_rect_windows = lambda hwnd: (383, 50, 703, 634)

        assert plugins._window_rect_windows(132628) == (383, 50, 703, 634)
    finally:
        plugins._dwm_window_rect_windows = original_dwm
        plugins._raw_window_rect_windows = original_raw
