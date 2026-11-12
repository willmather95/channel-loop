from __future__ import annotations

from channel_loop.cli import main

from conftest import BRIEF, GUARD, LOOPCFG

BASE = ["--loop-config", LOOPCFG, "--guard-config", GUARD, "--brief", BRIEF]


def test_init_tick_catch_up_verify_replay(tmp_path, capsys):
    s = str(tmp_path / "state")
    assert main(BASE + ["init", "--state", s, "--seed", "4", "--weeks", "2"]) == 0
    assert main(BASE + ["init", "--state", s]) == 1  # refuses to clobber
    assert main(BASE + ["tick", "--state", s, "--date", "2026-09-07"]) == 0
    assert main(BASE + ["tick", "--state", s, "--date", "2026-09-07"]) == 0  # no-op
    assert "noop" in capsys.readouterr().out
    assert main(BASE + ["tick", "--state", s, "--date", "2026-09-10"]) == 1  # gap: use catch-up
    assert main(BASE + ["catch-up", "--state", s, "--through", "2026-09-12"]) == 0
    assert main(BASE + ["journal", "verify", "--state", s]) == 0
    assert main(BASE + ["journal", "replay", "--state", s]) == 0
    out = capsys.readouterr().out
    assert "chain intact" in out and "MISMATCH" not in out
    assert main(BASE + ["journal", "show", "--state", s]) == 0


def test_kill_switch_exit_code(tmp_path, capsys):
    s = str(tmp_path / "state")
    main(BASE + ["init", "--state", s, "--seed", "4", "--weeks", "2"])
    assert main(BASE + ["kill", "on", "--state", s]) == 0
    assert main(BASE + ["catch-up", "--state", s, "--through", "2026-09-09"]) == 3
    assert main(BASE + ["kill", "off", "--state", s]) == 0
    assert main(BASE + ["catch-up", "--state", s, "--through", "2026-09-10"]) == 0


def test_serve_advances_one_day_per_wakeup(tmp_path, capsys):
    s = str(tmp_path / "state")
    main(BASE + ["init", "--state", s, "--seed", "4", "--weeks", "1"])
    assert main(BASE + ["serve", "--state", s, "--interval", "0", "--max-wakeups", "3"]) == 0
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("2026-")]
    assert [x[:10] for x in lines] == ["2026-09-07", "2026-09-08", "2026-09-09"]
