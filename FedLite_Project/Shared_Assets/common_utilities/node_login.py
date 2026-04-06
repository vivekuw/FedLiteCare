"""Lightweight local login prompt for FedLiteCare node GUIs."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

from FedLite_Project.Shared_Assets.common_utilities.common_utils import (
    load_simple_yaml_config,
)


def load_node_login_settings(config_path: Path) -> dict[str, Any]:
    """Load the flat per-node login settings from the node config."""
    settings = load_simple_yaml_config(config_path.resolve())
    hospital_name = str(settings.get("hospital_name", "Node"))
    default_username = hospital_name.lower()
    return {
        "login_enabled": bool(settings.get("login_enabled", False)),
        "login_username": str(settings.get("login_username", default_username)),
        "login_password": str(settings.get("login_password", "")),
        "login_window_title": str(
            settings.get("login_window_title", f"{hospital_name} Login")
        ),
        "login_subtitle": str(
            settings.get(
                "login_subtitle",
                "Sign in to open the local constrained-node client.",
            )
        ),
    }


class NodeLoginWindow(tk.Tk):
    """Small local login window for one node operator."""

    def __init__(self, login_settings: dict[str, Any]) -> None:
        super().__init__()
        self.login_settings = login_settings
        self.authenticated_username: str | None = None

        self.title(str(login_settings["login_window_title"]))
        self.geometry("420x230")
        self.minsize(380, 220)
        self.resizable(False, False)

        self.username_var = tk.StringVar(value="")
        self.password_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Enter node credentials")

        self._build_layout()
        self.bind("<Return>", self._attempt_login)
        self.protocol("WM_DELETE_WINDOW", self._cancel_login)

    def _build_layout(self) -> None:
        container = ttk.Frame(self, padding=18)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text=str(self.login_settings["login_window_title"]),
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            container,
            text=str(self.login_settings["login_subtitle"]),
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(6, 14))

        form = ttk.Frame(container)
        form.pack(fill="x")

        ttk.Label(form, text="Username").grid(row=0, column=0, sticky="w", pady=6)
        username_entry = ttk.Entry(form, textvariable=self.username_var, width=28)
        username_entry.grid(row=0, column=1, sticky="w", pady=6, padx=(10, 0))

        ttk.Label(form, text="Password").grid(row=1, column=0, sticky="w", pady=6)
        password_entry = ttk.Entry(
            form,
            textvariable=self.password_var,
            width=28,
            show="*",
        )
        password_entry.grid(row=1, column=1, sticky="w", pady=6, padx=(10, 0))

        actions = ttk.Frame(container)
        actions.pack(fill="x", pady=(16, 0))
        ttk.Button(actions, text="Sign In", command=self._attempt_login).pack(side="left")
        ttk.Button(actions, text="Cancel", command=self._cancel_login).pack(side="left", padx=(8, 0))

        ttk.Label(
            container,
            textvariable=self.status_var,
            foreground="#a1260d",
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(14, 0))

        username_entry.focus_set()
        password_entry.bind("<Return>", self._attempt_login)

    def _attempt_login(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        expected_username = str(self.login_settings["login_username"])
        expected_password = str(self.login_settings["login_password"])
        entered_username = self.username_var.get().strip()
        entered_password = self.password_var.get()

        if entered_username == expected_username and entered_password == expected_password:
            self.authenticated_username = entered_username
            self.destroy()
            return

        self.status_var.set("Invalid username or password. Please try again.")
        self.password_var.set("")

    def _cancel_login(self) -> None:
        self.authenticated_username = None
        self.destroy()


def request_node_login(config_path: Path) -> str | None:
    """Prompt for local node credentials when login is enabled."""
    login_settings = load_node_login_settings(config_path)
    if not login_settings["login_enabled"]:
        return str(login_settings["login_username"])

    login_window = NodeLoginWindow(login_settings)
    login_window.mainloop()
    return login_window.authenticated_username
