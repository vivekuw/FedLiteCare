"""Reusable Tkinter desktop client for FedLiteCare hospital nodes."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from queue import Empty, Queue
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Callable

from FedLite_Project.Shared_Assets.common_utilities.federated_hospital_node import (
    run_hospital_federated_round,
)
from FedLite_Project.Shared_Assets.common_utilities.hospital_gui_support import (
    copy_dataset_into_uploads,
    get_feature_columns_for_hospital,
    get_hospital_dashboard_status,
    list_available_dataset_files,
    read_recent_log_lines,
)
from FedLite_Project.Shared_Assets.common_utilities.hospital_quality_reports import (
    validate_training_dataset,
)
from FedLite_Project.Shared_Assets.common_utilities.local_ml_pipeline import (
    predict_from_patient_values,
    train_local_model,
)


class HospitalClientApp(tk.Tk):
    """Tkinter client that wraps the existing hospital backend modules."""

    def __init__(
        self,
        config_path: Path,
        receive_global_model_callable: Callable[..., dict[str, Any]],
        send_local_update_callable: Callable[..., dict[str, Any]],
    ) -> None:
        super().__init__()
        self.config_path = config_path.resolve()
        self.receive_global_model_callable = receive_global_model_callable
        self.send_local_update_callable = send_local_update_callable
        self.dashboard_status = get_hospital_dashboard_status(self.config_path)
        self.feature_columns = get_feature_columns_for_hospital(self.config_path)

        self.title(f"FedLiteCare Hospital Client - {self.dashboard_status['hospital_name']}")
        self.geometry("1120x780")
        self.minsize(980, 700)

        self.background_results: Queue[tuple[str, str, Any]] = Queue()
        self.busy_widgets: list[ttk.Widget] = []
        self.selected_upload_path: Path | None = None
        self.is_busy = False

        self.current_status_var = tk.StringVar(value="Ready")
        self.training_status_var = tk.StringVar(value="Idle")
        self.sync_status_var = tk.StringVar(value="Idle")
        self.validation_status_var = tk.StringVar(value="Not validated")
        self.upload_status_var = tk.StringVar(value="No dataset selected")
        self.prediction_result_var = tk.StringVar(value="No prediction yet")
        self.confidence_var = tk.StringVar(value="Confidence: N/A")
        self.validation_report_var = tk.StringVar(value="Validation report: N/A")
        self.prediction_report_var = tk.StringVar(value="Prediction report: N/A")
        self.model_version_var = tk.StringVar(value=self.dashboard_status["local_model_version"])
        self.global_version_var = tk.StringVar(value=self.dashboard_status["current_global_version"])
        self.training_accuracy_var = tk.StringVar(value=self._format_metric(self.dashboard_status["training_accuracy"]))
        self.training_loss_var = tk.StringVar(value=self._format_metric(self.dashboard_status["training_loss"]))
        self.model_path_var = tk.StringVar(value=str(self.dashboard_status["model_path"]))
        self.dataset_var = tk.StringVar(
            value=self.dashboard_status["active_dataset"]
            if self.dashboard_status["active_dataset"] in self.dashboard_status["dataset_files"]
            else ""
        )
        self.log_choice_var = tk.StringVar(value="training")
        self.upload_path_var = tk.StringVar(value="")
        self.field_vars = {column: tk.StringVar(value="") for column in self.feature_columns}

        self._configure_style()
        self._build_layout()
        self.refresh_dashboard()
        self.refresh_logs()
        self.after(150, self._process_background_results)

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
        ttk.Label(
            header,
            text=f"FedLiteCare Hospital Client",
            style="Header.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            header,
            text=f"Hospital ID: {self.dashboard_status['hospital_name']}",
            style="Subheader.TLabel",
        ).pack(anchor="w")

        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True)

        self.dashboard_tab = ttk.Frame(notebook, padding=12)
        self.upload_tab = ttk.Frame(notebook, padding=12)
        self.train_tab = ttk.Frame(notebook, padding=12)
        self.predict_tab = ttk.Frame(notebook, padding=12)
        self.sync_tab = ttk.Frame(notebook, padding=12)
        self.logs_tab = ttk.Frame(notebook, padding=12)

        notebook.add(self.dashboard_tab, text="Hospital Dashboard")
        notebook.add(self.upload_tab, text="Upload Dataset")
        notebook.add(self.train_tab, text="Train Local Model")
        notebook.add(self.predict_tab, text="Predict Patient Risk")
        notebook.add(self.sync_tab, text="Sync With Aggregator")
        notebook.add(self.logs_tab, text="Logs / Status")

        self._build_dashboard_tab()
        self._build_upload_tab()
        self._build_train_tab()
        self._build_predict_tab()
        self._build_sync_tab()
        self._build_logs_tab()

        footer = ttk.Frame(container, padding=(0, 8, 0, 0))
        footer.pack(fill="x")
        ttk.Label(footer, text="Status:", style="CardTitle.TLabel").pack(side="left")
        ttk.Label(footer, textvariable=self.current_status_var, style="Value.TLabel").pack(side="left", padx=(6, 0))

    def _build_dashboard_tab(self) -> None:
        top_row = ttk.Frame(self.dashboard_tab)
        top_row.pack(fill="x")

        self._build_value_card(top_row, "Hospital ID", self.dashboard_status["hospital_name"]).pack(
            side="left", fill="x", expand=True, padx=(0, 8)
        )
        self._build_value_card(top_row, "Selected Dataset", textvariable=self.dataset_var).pack(
            side="left", fill="x", expand=True, padx=8
        )
        self._build_value_card(top_row, "Local Model Version", textvariable=self.model_version_var).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )

        middle_row = ttk.Frame(self.dashboard_tab)
        middle_row.pack(fill="x", pady=(12, 0))
        self._build_value_card(middle_row, "Current Global Version", textvariable=self.global_version_var).pack(
            side="left", fill="x", expand=True, padx=(0, 8)
        )
        self._build_value_card(middle_row, "Validation Accuracy", textvariable=self.training_accuracy_var).pack(
            side="left", fill="x", expand=True, padx=8
        )
        self._build_value_card(middle_row, "Validation Loss", textvariable=self.training_loss_var).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )

        status_frame = ttk.LabelFrame(self.dashboard_tab, text="Runtime Status", padding=12)
        status_frame.pack(fill="x", pady=(16, 0))
        self._add_status_row(status_frame, 0, "Validation Status", self.validation_status_var)
        self._add_status_row(status_frame, 1, "Training Status", self.training_status_var)
        self._add_status_row(status_frame, 2, "Sync Status", self.sync_status_var)
        self._add_status_row(status_frame, 3, "Model Path", self.model_path_var)
        self._add_status_row(status_frame, 4, "Latest Validation Report", self.validation_report_var)
        self._add_status_row(status_frame, 5, "Latest Prediction Report", self.prediction_report_var)

        refresh_button = ttk.Button(
            self.dashboard_tab,
            text="Refresh Dashboard",
            command=self.refresh_dashboard,
        )
        refresh_button.pack(anchor="e", pady=(12, 0))
        self.busy_widgets.append(refresh_button)

    def _build_upload_tab(self) -> None:
        info = ttk.Label(
            self.upload_tab,
            text="Copy a CSV file into this hospital's uploads folder for training or local prediction.",
            wraplength=820,
            justify="left",
        )
        info.pack(anchor="w")

        controls = ttk.Frame(self.upload_tab)
        controls.pack(fill="x", pady=(14, 0))
        choose_button = ttk.Button(controls, text="Choose CSV File", command=self.choose_dataset_file)
        choose_button.pack(side="left")
        self.busy_widgets.append(choose_button)
        upload_button = ttk.Button(controls, text="Upload Into Hospital Folder", command=self.upload_selected_dataset)
        upload_button.pack(side="left", padx=(8, 0))
        self.busy_widgets.append(upload_button)

        ttk.Label(self.upload_tab, textvariable=self.upload_path_var).pack(anchor="w", pady=(14, 0))
        ttk.Label(self.upload_tab, textvariable=self.upload_status_var).pack(anchor="w", pady=(8, 0))

        dataset_frame = ttk.LabelFrame(self.upload_tab, text="Available Uploaded Datasets", padding=12)
        dataset_frame.pack(fill="both", expand=True, pady=(18, 0))
        self.dataset_listbox = tk.Listbox(dataset_frame, height=12)
        self.dataset_listbox.pack(fill="both", expand=True)
        self.dataset_listbox.bind("<<ListboxSelect>>", self._handle_dataset_selection)

    def _build_train_tab(self) -> None:
        form = ttk.Frame(self.train_tab)
        form.pack(fill="x")
        ttk.Label(form, text="Dataset to Train").grid(row=0, column=0, sticky="w")
        self.dataset_combo = ttk.Combobox(form, textvariable=self.dataset_var, state="readonly", width=40)
        self.dataset_combo.grid(row=1, column=0, sticky="w", pady=(4, 0))
        refresh_button = ttk.Button(form, text="Refresh Dataset List", command=self.refresh_dataset_list)
        refresh_button.grid(row=1, column=1, sticky="w", padx=(10, 0))
        self.busy_widgets.extend([self.dataset_combo, refresh_button])

        actions = ttk.Frame(self.train_tab)
        actions.pack(anchor="w", pady=(16, 0))
        validate_button = ttk.Button(actions, text="Validate Dataset", command=self.validate_selected_dataset)
        validate_button.pack(side="left")
        train_button = ttk.Button(actions, text="Train Local Model", command=self.train_selected_dataset)
        train_button.pack(side="left", padx=(8, 0))
        self.busy_widgets.extend([validate_button, train_button])
        ttk.Label(self.train_tab, textvariable=self.training_status_var).pack(anchor="w", pady=(10, 0))
        ttk.Label(self.train_tab, textvariable=self.validation_status_var).pack(anchor="w", pady=(6, 0))
        ttk.Label(self.train_tab, textvariable=self.validation_report_var).pack(anchor="w", pady=(6, 0))

    def _build_predict_tab(self) -> None:
        form = ttk.LabelFrame(self.predict_tab, text="Patient Features", padding=12)
        form.pack(fill="x")

        for index, column in enumerate(self.feature_columns):
            row = index // 2
            column_position = (index % 2) * 2
            ttk.Label(form, text=column).grid(row=row, column=column_position, sticky="w", padx=(0, 8), pady=6)
            ttk.Entry(form, textvariable=self.field_vars[column], width=22).grid(
                row=row,
                column=column_position + 1,
                sticky="w",
                pady=6,
            )

        actions = ttk.Frame(self.predict_tab)
        actions.pack(fill="x", pady=(16, 0))
        predict_button = ttk.Button(actions, text="Predict Patient Risk", command=self.predict_patient_risk)
        predict_button.pack(side="left")
        clear_button = ttk.Button(actions, text="Clear Fields", command=self.clear_prediction_fields)
        clear_button.pack(side="left", padx=(8, 0))
        self.busy_widgets.extend([predict_button, clear_button])

        result_frame = ttk.LabelFrame(self.predict_tab, text="Prediction Result", padding=12)
        result_frame.pack(fill="x", pady=(16, 0))
        ttk.Label(result_frame, textvariable=self.prediction_result_var, style="Value.TLabel").pack(anchor="w")
        ttk.Label(result_frame, textvariable=self.confidence_var, style="Value.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Label(result_frame, textvariable=self.prediction_report_var, style="Value.TLabel").pack(anchor="w", pady=(8, 0))

    def _build_sync_tab(self) -> None:
        info = ttk.Label(
            self.sync_tab,
            text=(
                "Use this action when the aggregator terminal is already running. "
                "The hospital client will wait for the current global model, train locally, and return the update."
            ),
            wraplength=840,
            justify="left",
        )
        info.pack(anchor="w")

        sync_button = ttk.Button(self.sync_tab, text="Sync With Aggregator", command=self.sync_with_aggregator)
        sync_button.pack(anchor="w", pady=(16, 0))
        self.busy_widgets.append(sync_button)
        ttk.Label(self.sync_tab, textvariable=self.sync_status_var).pack(anchor="w", pady=(10, 0))

    def _build_logs_tab(self) -> None:
        controls = ttk.Frame(self.logs_tab)
        controls.pack(fill="x")
        ttk.Label(controls, text="Log File").pack(side="left")
        self.log_combo = ttk.Combobox(
            controls,
            textvariable=self.log_choice_var,
            state="readonly",
            values=["training", "prediction", "sync", "transfer"],
            width=18,
        )
        self.log_combo.pack(side="left", padx=(8, 0))
        self.busy_widgets.append(self.log_combo)
        refresh_button = ttk.Button(controls, text="Refresh Logs", command=self.refresh_logs)
        refresh_button.pack(side="left", padx=(8, 0))
        self.busy_widgets.append(refresh_button)

        self.logs_text = ScrolledText(self.logs_tab, wrap="word", height=24, font=("Consolas", 10))
        self.logs_text.pack(fill="both", expand=True, pady=(12, 0))

    def _build_value_card(
        self,
        parent: ttk.Frame,
        title: str,
        value: str | None = None,
        textvariable: tk.StringVar | None = None,
    ) -> ttk.LabelFrame:
        card = ttk.LabelFrame(parent, text=title, padding=12)
        label = ttk.Label(card, text=value or "", textvariable=textvariable, style="Value.TLabel")
        label.pack(anchor="w")
        return card

    def _add_status_row(self, parent: ttk.Frame, row: int, title: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=title, style="CardTitle.TLabel").grid(row=row, column=0, sticky="w", pady=6)
        ttk.Label(parent, textvariable=variable, style="Value.TLabel").grid(
            row=row,
            column=1,
            sticky="w",
            padx=(12, 0),
            pady=6,
        )

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

    def _run_background_task(self, task_name: str, task_callable: Callable[[], Any]) -> None:
        if self.is_busy:
            messagebox.showinfo("Busy", "Another action is already running. Please wait for it to finish.")
            return

        self._set_busy(True)
        self.current_status_var.set(f"{task_name} is running...")

        def _worker() -> None:
            try:
                result = task_callable()
                self.background_results.put(("ok", task_name, result))
            except Exception as error:
                self.background_results.put(("error", task_name, error))

        threading.Thread(target=_worker, daemon=True).start()

    def _process_background_results(self) -> None:
        try:
            while True:
                status, task_name, payload = self.background_results.get_nowait()
                self._set_busy(False)
                if status == "ok":
                    self._handle_completed_task(task_name, payload)
                else:
                    self.current_status_var.set(f"{task_name} failed")
                    if task_name == "Train Local Model":
                        self.training_status_var.set("Training failed")
                    if task_name == "Validate Dataset":
                        self.validation_status_var.set("Validation failed")
                    if task_name == "Sync With Aggregator":
                        self.sync_status_var.set("Sync failed")
                    messagebox.showerror(task_name, str(payload))
                self.refresh_dashboard()
                self.refresh_logs()
        except Empty:
            pass

        self.after(150, self._process_background_results)

    def _handle_completed_task(self, task_name: str, payload: Any) -> None:
        self.current_status_var.set(f"{task_name} completed")

        if task_name == "Train Local Model":
            self.training_status_var.set(
                f"Training complete. Accuracy: {payload['validation_accuracy']:.4f}"
            )
            self.validation_status_var.set(f"Validation: {payload['validation_result']['status']}")
            self.validation_report_var.set(f"Validation report: {payload['validation_report_path'].name}")
            messagebox.showinfo(
                "Training Complete",
                (
                    f"Model saved to:\n{payload['model_path']}\n\n"
                    f"Validation: {payload['validation_result']['status']}\n"
                    f"Validation report: {payload['validation_report_path']}\n"
                    f"Validation accuracy: {payload['validation_accuracy']:.4f}"
                ),
            )
            return

        if task_name == "Validate Dataset":
            self.validation_status_var.set(f"Validation: {payload['status']}")
            self.validation_report_var.set(f"Validation report: {payload['report_path'].name}")
            self.current_status_var.set("Dataset validation completed")
            messagebox.showinfo(
                "Validation Complete",
                f"Status: {payload['status']}\nReport saved to:\n{payload['report_path']}",
            )
            return

        if task_name == "Predict Patient Risk":
            self.prediction_result_var.set(f"Prediction Result: {payload['result_label']}")
            self.confidence_var.set(f"Confidence Score: {payload['confidence_score']:.4f}")
            self.prediction_report_var.set(f"Prediction report: {payload['report_path'].name}")
            return

        if task_name == "Sync With Aggregator":
            self.sync_status_var.set(f"Sync complete for {payload['round_name']}")
            self.validation_status_var.set(
                f"Validation: {payload['training_result']['validation_result']['status']}"
            )
            self.validation_report_var.set(
                f"Validation report: {payload['training_result']['validation_report_path'].name}"
            )
            messagebox.showinfo(
                "Sync Complete",
                (
                    f"Completed {payload['round_name']}.\n"
                    f"Validation: {payload['training_result']['validation_result']['status']}\n"
                    f"Local update sent successfully."
                ),
            )
            return

    def refresh_dataset_list(self) -> None:
        datasets = list_available_dataset_files(self.config_path)
        self.dataset_combo["values"] = datasets
        self.dataset_listbox.delete(0, tk.END)
        for dataset_name in datasets:
            self.dataset_listbox.insert(tk.END, dataset_name)

        if datasets and self.dataset_var.get() not in datasets:
            self.dataset_var.set(datasets[0])
        if not datasets:
            self.dataset_var.set("")

    def refresh_dashboard(self) -> None:
        self.dashboard_status = get_hospital_dashboard_status(self.config_path)
        self.model_version_var.set(self.dashboard_status["local_model_version"])
        self.global_version_var.set(self.dashboard_status["current_global_version"])
        self.training_accuracy_var.set(self._format_metric(self.dashboard_status["training_accuracy"]))
        self.training_loss_var.set(self._format_metric(self.dashboard_status["training_loss"]))
        self.model_path_var.set(str(self.dashboard_status["model_path"]))
        latest_validation_report = self.dashboard_status["latest_validation_report"]
        latest_prediction_report = self.dashboard_status["latest_prediction_report"]
        latest_validation_status = self.dashboard_status["latest_validation_status"]
        if latest_validation_status:
            self.validation_status_var.set(f"Validation: {latest_validation_status}")
        self.validation_report_var.set(
            "Validation report: N/A" if latest_validation_report is None else f"Validation report: {latest_validation_report.name}"
        )
        self.prediction_report_var.set(
            "Prediction report: N/A" if latest_prediction_report is None else f"Prediction report: {latest_prediction_report.name}"
        )
        self.refresh_dataset_list()

    def refresh_logs(self) -> None:
        status = get_hospital_dashboard_status(self.config_path)
        selected_log = self.log_choice_var.get() or "training"
        log_path = status["logs"].get(selected_log)
        if log_path is None:
            return

        self.logs_text.delete("1.0", tk.END)
        self.logs_text.insert("1.0", read_recent_log_lines(log_path))

    def choose_dataset_file(self) -> None:
        selected_path = filedialog.askopenfilename(
            title="Choose Dataset CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not selected_path:
            return

        self.selected_upload_path = Path(selected_path)
        self.upload_path_var.set(f"Selected file: {self.selected_upload_path}")
        self.upload_status_var.set("Ready to upload")

    def upload_selected_dataset(self) -> None:
        if self.selected_upload_path is None:
            messagebox.showwarning("No File Selected", "Choose a CSV file first.")
            return

        destination_path = copy_dataset_into_uploads(self.config_path, self.selected_upload_path)
        self.upload_status_var.set(f"Dataset uploaded to: {destination_path.name}")
        self.dataset_var.set(destination_path.name)
        self.current_status_var.set("Dataset uploaded successfully")
        self.refresh_dashboard()

    def _handle_dataset_selection(self, event: tk.Event[tk.Listbox]) -> None:
        selection = self.dataset_listbox.curselection()
        if not selection:
            return
        self.dataset_var.set(self.dataset_listbox.get(selection[0]))

    def train_selected_dataset(self) -> None:
        dataset_name = self.dataset_var.get().strip() or None
        self.training_status_var.set("Training in progress...")
        self._run_background_task(
            "Train Local Model",
            lambda: train_local_model(
                config_path=self.config_path,
                dataset_filename=dataset_name,
            ),
        )

    def validate_selected_dataset(self) -> None:
        dataset_name = self.dataset_var.get().strip() or None
        self.validation_status_var.set("Validation in progress...")
        self._run_background_task(
            "Validate Dataset",
            lambda: validate_training_dataset(
                config_path=self.config_path,
                dataset_filename=dataset_name,
            ),
        )

    def clear_prediction_fields(self) -> None:
        for variable in self.field_vars.values():
            variable.set("")
        self.prediction_result_var.set("No prediction yet")
        self.confidence_var.set("Confidence: N/A")

    def predict_patient_risk(self) -> None:
        patient_values = {
            column: variable.get().strip()
            for column, variable in self.field_vars.items()
        }
        self.prediction_result_var.set("Running prediction...")
        self._run_background_task(
            "Predict Patient Risk",
            lambda: predict_from_patient_values(
                config_path=self.config_path,
                patient_values=patient_values,
            ),
        )

    def sync_with_aggregator(self) -> None:
        dataset_name = self.dataset_var.get().strip() or None
        self.sync_status_var.set("Waiting for aggregator and running federated round...")
        self._run_background_task(
            "Sync With Aggregator",
            lambda: run_hospital_federated_round(
                config_path=self.config_path,
                dataset_filename=dataset_name,
                receive_global_model_callable=self.receive_global_model_callable,
                send_local_update_callable=self.send_local_update_callable,
            ),
        )

    @staticmethod
    def _format_metric(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value:.4f}"


def launch_hospital_gui(
    config_path: Path,
    receive_global_model_callable: Callable[..., dict[str, Any]],
    send_local_update_callable: Callable[..., dict[str, Any]],
) -> None:
    """Launch the Tkinter hospital client."""
    app = HospitalClientApp(
        config_path=config_path,
        receive_global_model_callable=receive_global_model_callable,
        send_local_update_callable=send_local_update_callable,
    )
    app.mainloop()
