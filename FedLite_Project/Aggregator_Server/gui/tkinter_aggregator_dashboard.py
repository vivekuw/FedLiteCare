"""Tkinter dashboard for the FedLiteCare aggregator server."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from queue import Empty, Queue
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

from FedLite_Project.Aggregator_Server.server.aggregator_dashboard_support import (
    get_aggregator_dashboard_status,
    get_aggregator_log_text,
)
from FedLite_Project.Aggregator_Server.server.global_model_manager import (
    DEFAULT_CONFIG_PATH,
    HOSPITAL_ORDER,
    run_distributed_federated_round,
)


class AggregatorDashboardApp(tk.Tk):
    """Dashboard GUI for viewing node status and starting aggregator rounds."""

    LOG_NAMES = ("runtime", "aggregation", "round")

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        super().__init__()
        self.config_path = config_path.resolve()
        self.dashboard_status = get_aggregator_dashboard_status(self.config_path)
        self.background_events: Queue[tuple[str, str, Any]] = Queue()
        self.busy_widgets: list[ttk.Widget] = []
        self.is_busy = False

        self.title("FedLiteCare Aggregator Dashboard")
        self.geometry("1220x820")
        self.minsize(1040, 720)

        self.current_status_var = tk.StringVar(value="Aggregator ready")
        self.server_name_var = tk.StringVar(value=self.dashboard_status["server_name"])
        self.latest_round_var = tk.StringVar(value=self._format_round(self.dashboard_status["latest_completed_round"]))
        self.next_round_var = tk.StringVar(value=self._format_round(self.dashboard_status["next_round_number"]))
        self.latest_round_name_var = tk.StringVar(value=self.dashboard_status["latest_round_name"] or "round_000")
        self.latest_completed_at_var = tk.StringVar(value=self.dashboard_status["latest_round_completed_at"] or "N/A")
        self.global_model_var = tk.StringVar(value=self.dashboard_status["latest_global_model_path"].name)
        self.global_model_size_var = tk.StringVar(value=self.dashboard_status["latest_global_model_size_text"])
        self.total_versions_var = tk.StringVar(value=str(self.dashboard_status["total_saved_versions"]))
        self.startup_delay_var = tk.StringVar(value="8")
        self.log_choice_var = tk.StringVar(value="runtime")

        self.node_summary_vars: dict[str, dict[str, tk.StringVar]] = {}
        self.node_round_state_vars: dict[str, tk.StringVar] = {}
        for hospital_name in HOSPITAL_ORDER:
            self.node_summary_vars[hospital_name] = {
                "label": tk.StringVar(value=""),
                "readiness": tk.StringVar(value=""),
                "dataset": tk.StringVar(value=""),
                "rows": tk.StringVar(value=""),
                "local_version": tk.StringVar(value=""),
                "validation": tk.StringVar(value=""),
                "global_sync": tk.StringVar(value=""),
                "local_update": tk.StringVar(value=""),
                "received_update": tk.StringVar(value=""),
            }
            self.node_round_state_vars[hospital_name] = tk.StringVar(value="Idle")

        self._configure_style()
        self._build_layout()
        self.refresh_dashboard()
        self.refresh_logs()
        self.after(200, self._process_background_events)
        self.after(3000, self._periodic_refresh)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Subheader.TLabel", font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Value.TLabel", font=("Segoe UI", 11))

    def _build_layout(self) -> None:
        container = ttk.Frame(self, padding=14)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="FedLiteCare Aggregator Dashboard", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Server-side view of node readiness, LTX transfers, and federated round progress",
            style="Subheader.TLabel",
        ).pack(anchor="w")

        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True)
        self.overview_tab = ttk.Frame(notebook, padding=12)
        self.nodes_tab = ttk.Frame(notebook, padding=12)
        self.round_tab = ttk.Frame(notebook, padding=12)
        self.logs_tab = ttk.Frame(notebook, padding=12)
        notebook.add(self.overview_tab, text="Overview")
        notebook.add(self.nodes_tab, text="Nodes")
        notebook.add(self.round_tab, text="Run Round")
        notebook.add(self.logs_tab, text="Logs")

        self._build_overview_tab()
        self._build_nodes_tab()
        self._build_round_tab()
        self._build_logs_tab()

        footer = ttk.Frame(container, padding=(0, 8, 0, 0))
        footer.pack(fill="x")
        ttk.Label(footer, text="Status:", style="CardTitle.TLabel").pack(side="left")
        ttk.Label(footer, textvariable=self.current_status_var, style="Value.TLabel").pack(side="left", padx=(6, 0))

    def _build_value_card(self, parent: ttk.Frame, title: str, variable: tk.StringVar) -> ttk.LabelFrame:
        card = ttk.LabelFrame(parent, text=title, padding=12)
        ttk.Label(card, textvariable=variable, style="Value.TLabel").pack(anchor="w")
        return card

    def _add_status_row(self, parent: ttk.Frame, row: int, title: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=title, style="CardTitle.TLabel").grid(row=row, column=0, sticky="w", pady=6)
        ttk.Label(parent, textvariable=variable, style="Value.TLabel", wraplength=620, justify="left").grid(
            row=row,
            column=1,
            sticky="w",
            padx=(12, 0),
            pady=6,
        )

    def _build_overview_tab(self) -> None:
        first_row = ttk.Frame(self.overview_tab)
        first_row.pack(fill="x")
        self._build_value_card(first_row, "Server Name", self.server_name_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._build_value_card(first_row, "Latest Completed Round", self.latest_round_var).pack(side="left", fill="x", expand=True, padx=8)
        self._build_value_card(first_row, "Next Round", self.next_round_var).pack(side="left", fill="x", expand=True, padx=(8, 0))

        second_row = ttk.Frame(self.overview_tab)
        second_row.pack(fill="x", pady=(12, 0))
        self._build_value_card(second_row, "Latest Round Name", self.latest_round_name_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._build_value_card(second_row, "Latest Global Model", self.global_model_var).pack(side="left", fill="x", expand=True, padx=8)
        self._build_value_card(second_row, "Global Model Size", self.global_model_size_var).pack(side="left", fill="x", expand=True, padx=(8, 0))

        summary = ttk.LabelFrame(self.overview_tab, text="Aggregator Summary", padding=12)
        summary.pack(fill="x", pady=(18, 0))
        self._add_status_row(summary, 0, "Current Status", self.current_status_var)
        self._add_status_row(summary, 1, "Latest Completed At", self.latest_completed_at_var)
        self._add_status_row(summary, 2, "Saved Model Versions", self.total_versions_var)

        refresh_button = ttk.Button(self.overview_tab, text="Refresh Dashboard", command=self.refresh_dashboard)
        refresh_button.pack(anchor="e", pady=(12, 0))
        self.busy_widgets.append(refresh_button)

    def _build_nodes_tab(self) -> None:
        intro = ttk.Label(
            self.nodes_tab,
            text="Each node card shows dataset readiness, local model state, latest sync info, and current round state.",
            wraplength=980,
            justify="left",
        )
        intro.pack(anchor="w")

        cards_container = ttk.Frame(self.nodes_tab)
        cards_container.pack(fill="both", expand=True, pady=(16, 0))

        for index, hospital_name in enumerate(HOSPITAL_ORDER):
            card = ttk.LabelFrame(cards_container, text=hospital_name.replace("Hospital_", "Node "), padding=12)
            card.grid(row=0, column=index, sticky="nsew", padx=8)
            cards_container.columnconfigure(index, weight=1)
            vars_for_node = self.node_summary_vars[hospital_name]
            self._add_status_row(card, 0, "Readiness", vars_for_node["readiness"])
            self._add_status_row(card, 1, "Round State", self.node_round_state_vars[hospital_name])
            self._add_status_row(card, 2, "Dataset", vars_for_node["dataset"])
            self._add_status_row(card, 3, "Rows", vars_for_node["rows"])
            self._add_status_row(card, 4, "Local Model", vars_for_node["local_version"])
            self._add_status_row(card, 5, "Validation", vars_for_node["validation"])
            self._add_status_row(card, 6, "Global Sync", vars_for_node["global_sync"])
            self._add_status_row(card, 7, "Latest Local Update", vars_for_node["local_update"])
            self._add_status_row(card, 8, "Received By Server", vars_for_node["received_update"])

    def _build_round_tab(self) -> None:
        ttk.Label(
            self.round_tab,
            text=(
                "Start one distributed federated round from the dashboard. "
                "Nodes train from their cached global model, send local updates to the server, and then receive "
                "the refreshed global model at the end of the round. The server will only bootstrap nodes that "
                "do not already have a cached global checkpoint."
            ),
            wraplength=980,
            justify="left",
        ).pack(anchor="w")

        controls = ttk.LabelFrame(self.round_tab, text="Round Controls", padding=12)
        controls.pack(fill="x", pady=(16, 0))
        ttk.Label(controls, text="Startup Delay (seconds)").grid(row=0, column=0, sticky="w")
        self.startup_delay_entry = ttk.Entry(controls, textvariable=self.startup_delay_var, width=10)
        self.startup_delay_entry.grid(row=0, column=1, sticky="w", padx=(8, 0))
        start_button = ttk.Button(controls, text="Start Distributed Round", command=self.start_round)
        start_button.grid(row=0, column=2, sticky="w", padx=(14, 0))
        refresh_button = ttk.Button(controls, text="Refresh Status", command=self.refresh_dashboard)
        refresh_button.grid(row=0, column=3, sticky="w", padx=(8, 0))
        self.busy_widgets.extend([self.startup_delay_entry, start_button, refresh_button])

        ttk.Label(
            self.round_tab,
            text="Live Round Progress",
            style="CardTitle.TLabel",
        ).pack(anchor="w", pady=(18, 0))
        self.progress_text = ScrolledText(self.round_tab, wrap="word", height=24, font=("Consolas", 10))
        self.progress_text.pack(fill="both", expand=True, pady=(8, 0))

    def _build_logs_tab(self) -> None:
        controls = ttk.Frame(self.logs_tab)
        controls.pack(fill="x")
        ttk.Label(controls, text="Log Stream").pack(side="left")
        self.log_combo = ttk.Combobox(
            controls,
            textvariable=self.log_choice_var,
            state="readonly",
            values=list(self.LOG_NAMES),
            width=16,
        )
        self.log_combo.pack(side="left", padx=(8, 0))
        refresh_button = ttk.Button(controls, text="Refresh Logs", command=self.refresh_logs)
        refresh_button.pack(side="left", padx=(8, 0))
        self.busy_widgets.extend([self.log_combo, refresh_button])

        self.logs_text = ScrolledText(self.logs_tab, wrap="word", height=30, font=("Consolas", 10))
        self.logs_text.pack(fill="both", expand=True, pady=(12, 0))

    def _set_busy(self, busy: bool) -> None:
        self.is_busy = busy
        for widget in self.busy_widgets:
            try:
                if isinstance(widget, ttk.Combobox):
                    widget.configure(state="disabled" if busy else "readonly")
                else:
                    widget.configure(state="disabled" if busy else "normal")
            except tk.TclError:
                continue

    def refresh_dashboard(self) -> None:
        self.dashboard_status = get_aggregator_dashboard_status(self.config_path)
        self.server_name_var.set(self.dashboard_status["server_name"])
        self.latest_round_var.set(self._format_round(self.dashboard_status["latest_completed_round"]))
        self.next_round_var.set(self._format_round(self.dashboard_status["next_round_number"]))
        self.latest_round_name_var.set(self.dashboard_status["latest_round_name"] or "round_000")
        self.latest_completed_at_var.set(self.dashboard_status["latest_round_completed_at"] or "N/A")
        self.global_model_var.set(self.dashboard_status["latest_global_model_path"].name)
        self.global_model_size_var.set(self.dashboard_status["latest_global_model_size_text"])
        self.total_versions_var.set(str(self.dashboard_status["total_saved_versions"]))

        for hospital_name in HOSPITAL_ORDER:
            node_status = self.dashboard_status["nodes"][hospital_name]
            node_vars = self.node_summary_vars[hospital_name]
            node_vars["readiness"].set(node_status["readiness"])
            node_vars["dataset"].set(node_status["active_dataset"])
            node_vars["rows"].set(self._format_rows(node_status["dataset_row_count"]))
            node_vars["local_version"].set(node_status["local_model_version"])
            node_vars["validation"].set(node_status["latest_validation_status"] or "Not checked")
            node_vars["global_sync"].set(node_status["current_global_version"])
            node_vars["local_update"].set(self._format_node_file(node_status["latest_local_update_path"]))
            node_vars["received_update"].set(self._format_node_file(node_status["latest_received_update_path"]))

    def refresh_logs(self) -> None:
        log_name = self.log_choice_var.get() or "runtime"
        self.logs_text.delete("1.0", tk.END)
        self.logs_text.insert("1.0", get_aggregator_log_text(self.config_path, log_name=log_name))

    def start_round(self) -> None:
        if self.is_busy:
            messagebox.showinfo("Busy", "A round is already running.")
            return

        try:
            startup_delay_seconds = max(int(self.startup_delay_var.get().strip() or "0"), 0)
        except ValueError:
            messagebox.showerror("Invalid Delay", "Startup delay must be a whole number.")
            return

        for hospital_name in HOSPITAL_ORDER:
            self.node_round_state_vars[hospital_name].set("Waiting for round start")
        self.progress_text.delete("1.0", tk.END)
        self._set_busy(True)
        self.current_status_var.set("Distributed round is starting...")

        def _progress_callback(message: str) -> None:
            self.background_events.put(("progress", "round", message))

        def _worker() -> None:
            try:
                result = run_distributed_federated_round(
                    config_path=self.config_path,
                    progress_callback=_progress_callback,
                    wait_for_hospital_confirmation=False,
                    startup_delay_seconds=startup_delay_seconds,
                )
                self.background_events.put(("ok", "round", result))
            except Exception as error:
                self.background_events.put(("error", "round", error))

        threading.Thread(target=_worker, daemon=True).start()

    def _process_background_events(self) -> None:
        try:
            while True:
                event_type, task_name, payload = self.background_events.get_nowait()
                if event_type == "progress":
                    self._append_progress_line(str(payload))
                    self._apply_progress_message(str(payload))
                elif event_type == "ok":
                    self._set_busy(False)
                    self._handle_round_complete(payload)
                    self.refresh_dashboard()
                    self.refresh_logs()
                elif event_type == "error":
                    self._set_busy(False)
                    self.current_status_var.set("Distributed round failed")
                    for hospital_name in HOSPITAL_ORDER:
                        if self.node_round_state_vars[hospital_name].get() != "Update received":
                            self.node_round_state_vars[hospital_name].set("Round failed")
                    messagebox.showerror("Distributed Round Failed", str(payload))
                    self.refresh_dashboard()
                    self.refresh_logs()
        except Empty:
            pass

        self.after(200, self._process_background_events)

    def _append_progress_line(self, message: str) -> None:
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)

    def _apply_progress_message(self, message: str) -> None:
        self.current_status_var.set(message)
        if "listeners are ready" in message.lower():
            for hospital_name in HOSPITAL_ORDER:
                self.node_round_state_vars[hospital_name].set("Waiting for local update")
        if "bootstrap global model" in message.lower():
            for hospital_name in HOSPITAL_ORDER:
                if hospital_name in message:
                    self.node_round_state_vars[hospital_name].set("Bootstrap model sent")
        if "performing federated averaging" in message.lower():
            for hospital_name in HOSPITAL_ORDER:
                if self.node_round_state_vars[hospital_name].get() == "Update received":
                    self.node_round_state_vars[hospital_name].set("Ready for aggregation")
        for hospital_name in HOSPITAL_ORDER:
            if "refreshed global model" in message.lower() and hospital_name in message:
                self.node_round_state_vars[hospital_name].set("Refreshed global sent")
            if f"update received" in message.lower() and hospital_name in message:
                self.node_round_state_vars[hospital_name].set("Update received")

    def _handle_round_complete(self, result: dict[str, Any]) -> None:
        completed_round = result["round_name"]
        self.current_status_var.set(f"Distributed round complete: {completed_round}")
        for hospital_name in HOSPITAL_ORDER:
            self.node_round_state_vars[hospital_name].set("Round completed")
        self._append_progress_line("------------------------------------------------")
        self._append_progress_line(f"Completed round: {completed_round}")
        self._append_progress_line(f"Latest global model: {result['latest_model_path']}")
        messagebox.showinfo(
            "Distributed Round Complete",
            (
                f"Completed {completed_round}.\n\n"
                f"Latest global model:\n{result['latest_model_path']}\n\n"
                f"Round log:\n{result['round_log_path']}"
            ),
        )

    def _periodic_refresh(self) -> None:
        if not self.is_busy:
            self.refresh_dashboard()
            self.refresh_logs()
        self.after(3000, self._periodic_refresh)

    @staticmethod
    def _format_round(round_number: int) -> str:
        return f"round_{int(round_number):03d}"

    @staticmethod
    def _format_rows(row_count: int | None) -> str:
        return "Unknown" if row_count is None else str(row_count)

    @staticmethod
    def _format_node_file(path_value: Path | None) -> str:
        return "N/A" if path_value is None else path_value.name


def launch_aggregator_dashboard(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Launch the Tkinter aggregator dashboard."""
    app = AggregatorDashboardApp(config_path=config_path)
    app.mainloop()
