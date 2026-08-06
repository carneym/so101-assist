"""Live tuning GUI for the SO-101 teleop.

Sliders for servo speeds, the lift-torque settings (setpoint leash +
driver step clamp), the load guard, and each joint's operating range.
Every change is written to config/tuning.json; a running
scripts/quadstick_teleop.py watches that file and applies the new
values to the arm within a fraction of a second — so tune while you
drive, no restart.

Run it in a second terminal alongside teleop:

    python scripts/tuning_gui.py

The workspace-fence panel also shows a live per-axis readout of the
end-effector position (including Z), published by a running teleop, so
you can set the fence min/max against where the arm actually is. It
reads "live —" when teleop isn't publishing.

"Reset to config baseline" clears the override (reverts to
config/default.yaml). Values persist in config/tuning.json across runs.
"""
from __future__ import annotations

import argparse
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import yaml

from so101_assist.control import tuning as T

SAVE_DEBOUNCE_MS = 150
STATUS_POLL_MS = 200      # how often to re-read the live EE position
STATUS_STALE_S = 1.5      # older than this = teleop not publishing


class TuningGUI:
    def __init__(
        self, root: tk.Tk, cfg: dict, path: Path, status_path: Path = T.DEFAULT_STATUS_PATH
    ) -> None:
        self.root = root
        self.cfg = cfg
        self.path = path
        self.status_path = status_path
        self._save_job: str | None = None

        resolved = T.resolve(cfg, T.load(path))
        self.scalar_vars: dict[str, tk.DoubleVar] = {}
        self.limit_vars: dict[str, tuple[tk.DoubleVar, tk.DoubleVar]] = {}
        self.fence_vars: dict[str, tuple[tk.DoubleVar, tk.DoubleVar]] = {}
        self.fence_live: dict[str, tk.StringVar] = {}   # live EE readout per axis
        self.load_enabled = tk.BooleanVar(value=resolved["load_stop_threshold"] is not None)

        root.title("SO-101 live tuning")
        left = ttk.LabelFrame(root, text="Speeds / torque / load guard", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        mid = ttk.LabelFrame(root, text="Joint operating range (deg)", padding=8)
        mid.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        right = ttk.LabelFrame(root, text="Workspace fence (m, base frame)", padding=8)
        right.grid(row=0, column=2, sticky="nsew", padx=6, pady=6)

        self._build_scalars(left, resolved)
        self._build_limits(mid, resolved)
        self._build_fence(right, resolved)

        btns = ttk.Frame(root, padding=6)
        btns.grid(row=1, column=0, columnspan=3, sticky="ew")
        ttk.Button(btns, text="Reset to config baseline", command=self._reset).pack(side="left")
        self.status = ttk.Label(btns, text=f"writing {path}")
        self.status.pack(side="right")

        self._poll_status()

    def _build_scalars(self, parent: ttk.Frame, resolved: dict) -> None:
        for row, (key, (label, lo, hi, step)) in enumerate(T.SCALARS.items()):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
            if key == "load_stop_threshold":
                ttk.Checkbutton(
                    parent, text="enabled", variable=self.load_enabled, command=self._changed
                ).grid(row=row, column=2, sticky="e")
            value = resolved[key]
            if value is None:   # disabled load guard — park slider mid-range
                value = (lo + hi) / 2
            var = tk.DoubleVar(value=value)
            self.scalar_vars[key] = var
            tk.Scale(
                parent, from_=lo, to=hi, resolution=step, orient="horizontal",
                variable=var, command=lambda _v: self._changed(), length=240,
            ).grid(row=row, column=1, sticky="ew")

    def _build_limits(self, parent: ttk.Frame, resolved: dict) -> None:
        lo_b, hi_b, step = T.JOINT_LIMIT_SLIDER
        row = 0
        for joint in T.JOINTS:
            j_lo, j_hi = resolved["joint_limits_deg"][joint]
            ttk.Label(parent, text=joint).grid(row=row, column=0, columnspan=2, sticky="w")
            row += 1
            lo_var, hi_var = tk.DoubleVar(value=j_lo), tk.DoubleVar(value=j_hi)
            self.limit_vars[joint] = (lo_var, hi_var)
            ttk.Label(parent, text="min").grid(row=row, column=0, sticky="e")
            tk.Scale(
                parent, from_=lo_b, to=hi_b, resolution=step, orient="horizontal",
                variable=lo_var, command=lambda _v: self._changed(), length=200,
            ).grid(row=row, column=1, sticky="ew")
            row += 1
            ttk.Label(parent, text="max").grid(row=row, column=0, sticky="e")
            tk.Scale(
                parent, from_=lo_b, to=hi_b, resolution=step, orient="horizontal",
                variable=hi_var, command=lambda _v: self._changed(), length=200,
            ).grid(row=row, column=1, sticky="ew")
            row += 1

    def _build_fence(self, parent: ttk.Frame, resolved: dict) -> None:
        lo_b, hi_b, step = T.FENCE_SLIDER
        row = 0
        for ax in T.FENCE_AXES:
            a_lo, a_hi = resolved["workspace_fence"][ax]
            live = tk.StringVar(value="live —")
            self.fence_live[ax] = live
            ttk.Label(parent, text=f"{ax}  (m)").grid(row=row, column=0, sticky="w")
            # Live EE position for this axis, published by a running
            # teleop — tune the min/max against where the arm actually is.
            ttk.Label(parent, textvariable=live).grid(row=row, column=1, sticky="e")
            row += 1
            lo_var, hi_var = tk.DoubleVar(value=a_lo), tk.DoubleVar(value=a_hi)
            self.fence_vars[ax] = (lo_var, hi_var)
            for tag, var in (("min", lo_var), ("max", hi_var)):
                ttk.Label(parent, text=tag).grid(row=row, column=0, sticky="e")
                tk.Scale(
                    parent, from_=lo_b, to=hi_b, resolution=step, orient="horizontal",
                    variable=var, command=lambda _v: self._changed(), length=200,
                ).grid(row=row, column=1, sticky="ew")
                row += 1

    def collect(self) -> dict:
        values: dict = {}
        for key in T.SCALARS:
            values[key] = self.scalar_vars[key].get()
        values["load_stop_ticks"] = round(values["load_stop_ticks"])
        values["max_joint_step_deg"] = round(values["max_joint_step_deg"])
        if not self.load_enabled.get():
            values["load_stop_threshold"] = None
        values["joint_limits_deg"] = {
            joint: [round(lo.get()), round(hi.get())]
            for joint, (lo, hi) in self.limit_vars.items()
        }
        values["workspace_fence"] = {
            ax: [round(lo.get(), 3), round(hi.get(), 3)]
            for ax, (lo, hi) in self.fence_vars.items()
        }
        return values

    def _changed(self) -> None:
        if self._save_job is not None:
            self.root.after_cancel(self._save_job)
        self._save_job = self.root.after(SAVE_DEBOUNCE_MS, self._save)

    def _save(self) -> None:
        self._save_job = None
        T.save(self.collect(), self.path)
        self.status.config(text=f"saved {self.path}")

    def _reset(self) -> None:
        T.save({}, self.path)
        resolved = T.resolve(self.cfg, {})
        for key, var in self.scalar_vars.items():
            v = resolved[key]
            var.set(v if v is not None else var.get())
        self.load_enabled.set(resolved["load_stop_threshold"] is not None)
        for joint, (lo, hi) in self.limit_vars.items():
            j_lo, j_hi = resolved["joint_limits_deg"][joint]
            lo.set(j_lo)
            hi.set(j_hi)
        for ax, (lo, hi) in self.fence_vars.items():
            a_lo, a_hi = resolved["workspace_fence"][ax]
            lo.set(a_lo)
            hi.set(a_hi)
        self.status.config(text="reset to config baseline")

    def _poll_status(self) -> None:
        """Refresh the per-axis live EE readout from the status file a
        running teleop publishes. Shows a dash when the reading is stale
        (teleop not running), so a frozen number is never mistaken for a
        live one. Re-arms itself on the Tk event loop."""
        status = T.read_status(self.status_path)
        ee = status.get("ee_xyz")
        stamp = status.get("stamp")
        fresh = ee is not None and stamp is not None and (time.time() - stamp) < STATUS_STALE_S
        for i, ax in enumerate(T.FENCE_AXES):
            self.fence_live[ax].set(f"live {ee[i]:+.3f}" if fresh else "live —")
        self.root.after(STATUS_POLL_MS, self._poll_status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--tuning", type=Path, default=T.DEFAULT_TUNING_PATH)
    parser.add_argument("--status", type=Path, default=T.DEFAULT_STATUS_PATH)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    root = tk.Tk()
    TuningGUI(root, cfg, args.tuning, args.status)
    root.mainloop()


if __name__ == "__main__":
    main()
