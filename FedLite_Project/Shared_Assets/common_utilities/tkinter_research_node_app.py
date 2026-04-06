"""Simplified research-demo GUI for FedLiteCare node simulations."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from queue import Empty, Queue
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Callable

from FedLite_Project.Shared_Assets.common_utilities.federated_hospital_node import run_hospital_federated_round
from FedLite_Project.Shared_Assets.common_utilities.hospital_gui_support import (
    copy_dataset_into_uploads,
    get_research_node_status,
    list_available_dataset_files,
    read_recent_log_lines,
)
from FedLite_Project.Shared_Assets.common_utilities.hospital_quality_reports import validate_training_dataset
from FedLite_Project.Shared_Assets.common_utilities.local_ml_pipeline import (
    load_hospital_context,
    predict_from_csv,
    train_local_model,
)
from FedLite_Project.Shared_Assets.data_preprocessing_helpers.preprocessing_utils import load_csv_records


class ResearchNodeApp(tk.Tk):
    """Node-focused desktop client for the constrained-environment demo."""

    LOG_LABEL_TO_KEY = {
        "training": "training",
        "evaluation": "prediction",
        "sync": "sync",
        "transfer": "transfer",
    }

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
        self.node_status = get_research_node_status(self.config_path)
        self.selected_upload_path: Path | None = None
        self.is_busy = False
        self.busy_widgets: list[ttk.Widget] = []
        self.background_results: Queue[tuple[str, str, Any]] = Queue()

        self.title(f"FedLiteCare Research Node - {self.node_status['node_label']}")
        self.geometry("1160x780")
        self.minsize(980, 700)

        self.current_status_var = tk.StringVar(value="Ready for constrained-node simulation")
        self.dataset_var = tk.StringVar(value=self.node_status["active_dataset"])
        self.node_label_var = tk.StringVar(value=self.node_status["node_label"])
        self.hospital_id_var = tk.StringVar(value=self.node_status["hospital_name"])
        self.dataset_path_var = tk.StringVar(value=str(self.node_status["dataset_path"]))
        self.dataset_rows_var = tk.StringVar(value=self._format_count(self.node_status["dataset_row_count"]))
        self.dataset_file_count_var = tk.StringVar(value=str(self.node_status["dataset_file_count"]))
        self.local_model_version_var = tk.StringVar(value=self.node_status["local_model_version"])
        self.global_model_version_var = tk.StringVar(value=self.node_status["current_global_version"])
        self.model_size_var = tk.StringVar(value=self.node_status["model_size_text"])
        self.latest_global_size_var = tk.StringVar(value=self.node_status["latest_global_model_size_text"])
        self.latest_update_size_var = tk.StringVar(value=self.node_status["latest_local_update_size_text"])
        self.latest_global_model_var = tk.StringVar(value=self._format_path_label(self.node_status["latest_global_model_path"]))
        self.latest_local_update_var = tk.StringVar(value=self._format_path_label(self.node_status["latest_local_update_path"]))
        self.training_accuracy_var = tk.StringVar(value=self._format_metric(self.node_status["training_accuracy"]))
        self.training_loss_var = tk.StringVar(value=self._format_metric(self.node_status["training_loss"]))
        self.validation_status_var = tk.StringVar(value=self._format_validation_status(self.node_status["latest_validation_status"]))
        self.validation_report_var = tk.StringVar(value=self._format_path_label(self.node_status["latest_validation_report"]))
        self.training_status_var = tk.StringVar(value="No local adaptation run yet")
        self.sync_status_var = tk.StringVar(value="No round synchronization yet")
        self.evaluation_status_var = tk.StringVar(value="No batch evaluation run yet")
        self.evaluation_output_var = tk.StringVar(value=self._format_path_label(self.node_status["latest_evaluation_output_path"]))
        self.upload_status_var = tk.StringVar(value="No dataset file selected")
        self.upload_path_var = tk.StringVar(value="")
        self.log_choice_var = tk.StringVar(value="training")

        self._configure_style()
        self._build_layout()
        self.refresh_node_status()
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
        ttk.Label(header, text="FedLiteCare Research Demo", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Lightweight model transfer and local adaptation on constrained nodes",
            style="Subheader.TLabel",
        ).pack(anchor="w")

        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True)
        self.overview_tab = ttk.Frame(notebook, padding=12)
        self.node_status_tab = ttk.Frame(notebook, padding=12)
        self.run_round_tab = ttk.Frame(notebook, padding=12)
        self.results_tab = ttk.Frame(notebook, padding=12)
        notebook.add(self.overview_tab, text="Overview")
        notebook.add(self.node_status_tab, text="Node Status")
        notebook.add(self.run_round_tab, text="Run Round")
        notebook.add(self.results_tab, text="Results")

        self._build_overview_tab()
        self._build_node_status_tab()
        self._build_run_round_tab()
        self._build_results_tab()

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
        ttk.Label(parent, textvariable=variable, style="Value.TLabel", wraplength=560, justify="left").grid(
            row=row,
            column=1,
            sticky="w",
            padx=(12, 0),
            pady=6,
        )

    def _build_overview_tab(self) -> None:
        ttk.Label(
            self.overview_tab,
            text=(
                "This interface is focused on the constrained-environment experiment. "
                "Select a local dataset, validate it, run local adaptation, exchange model "
                "updates through LTX, and inspect lightweight round results."
            ),
            wraplength=920,
            justify="left",
        ).pack(anchor="w")

        row_one = ttk.Frame(self.overview_tab)
        row_one.pack(fill="x", pady=(16, 0))
        self._build_value_card(row_one, "Node Label", self.node_label_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._build_value_card(row_one, "Source ID", self.hospital_id_var).pack(side="left", fill="x", expand=True, padx=8)
        self._build_value_card(row_one, "Selected Dataset", self.dataset_var).pack(side="left", fill="x", expand=True, padx=(8, 0))

        row_two = ttk.Frame(self.overview_tab)
        row_two.pack(fill="x", pady=(12, 0))
        self._build_value_card(row_two, "Local Model Version", self.local_model_version_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._build_value_card(row_two, "Global Model Version", self.global_model_version_var).pack(side="left", fill="x", expand=True, padx=8)
        self._build_value_card(row_two, "Dataset Rows", self.dataset_rows_var).pack(side="left", fill="x", expand=True, padx=(8, 0))

        row_three = ttk.Frame(self.overview_tab)
        row_three.pack(fill="x", pady=(12, 0))
        self._build_value_card(row_three, "Local Model Size", self.model_size_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._build_value_card(row_three, "Latest Global File Size", self.latest_global_size_var).pack(side="left", fill="x", expand=True, padx=8)
        self._build_value_card(row_three, "Latest Update File Size", self.latest_update_size_var).pack(side="left", fill="x", expand=True, padx=(8, 0))

        summary = ttk.LabelFrame(self.overview_tab, text="Experiment Summary", padding=12)
        summary.pack(fill="x", pady=(16, 0))
        self._add_status_row(summary, 0, "Current Status", self.current_status_var)
        self._add_status_row(summary, 1, "Dataset Validation", self.validation_status_var)
        self._add_status_row(summary, 2, "Local Adaptation", self.training_status_var)
        self._add_status_row(summary, 3, "Round Synchronization", self.sync_status_var)
        self._add_status_row(summary, 4, "Latest Evaluation Output", self.evaluation_output_var)

        refresh_button = ttk.Button(self.overview_tab, text="Refresh Overview", command=self.refresh_node_status)
        refresh_button.pack(anchor="e", pady=(12, 0))
        self.busy_widgets.append(refresh_button)

    def _build_node_status_tab(self) -> None:
        ttk.Label(
            self.node_status_tab,
            text=(
                "Manage the local CSV files used by this simulated node. "
                "Import datasets, select the active file, and inspect the current node artifacts."
            ),
            wraplength=920,
            justify="left",
        ).pack(anchor="w")

        controls = ttk.Frame(self.node_status_tab)
        controls.pack(fill="x", pady=(14, 0))
        import_button = ttk.Button(controls, text="Import CSV", command=self.choose_dataset_file)
        copy_button = ttk.Button(controls, text="Copy Into Node Folder", command=self.upload_selected_dataset)
        refresh_button = ttk.Button(controls, text="Refresh Dataset List", command=self.refresh_node_status)
        import_button.pack(side="left")
        copy_button.pack(side="left", padx=(8, 0))
        refresh_button.pack(side="left", padx=(8, 0))
        self.busy_widgets.extend([import_button, copy_button, refresh_button])

        ttk.Label(self.node_status_tab, textvariable=self.upload_path_var).pack(anchor="w", pady=(12, 0))
        ttk.Label(self.node_status_tab, textvariable=self.upload_status_var).pack(anchor="w", pady=(6, 0))

        select_frame = ttk.Frame(self.node_status_tab)
        select_frame.pack(fill="x", pady=(18, 0))
        ttk.Label(select_frame, text="Active Dataset", style="CardTitle.TLabel").pack(anchor="w")
        self.dataset_combo = ttk.Combobox(select_frame, textvariable=self.dataset_var, state="readonly", width=44)
        self.dataset_combo.pack(anchor="w", pady=(6, 0))
        self.dataset_combo.bind("<<ComboboxSelected>>", self._handle_dataset_combo_selection)
        self.busy_widgets.append(self.dataset_combo)

        lower = ttk.Frame(self.node_status_tab)
        lower.pack(fill="both", expand=True, pady=(18, 0))
        files_frame = ttk.LabelFrame(lower, text="Node CSV Files", padding=12)
        files_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.dataset_listbox = tk.Listbox(files_frame, height=14)
        self.dataset_listbox.pack(fill="both", expand=True)
        self.dataset_listbox.bind("<<ListboxSelect>>", self._handle_dataset_selection)

        details = ttk.LabelFrame(lower, text="Selected Dataset Details", padding=12)
        details.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._add_status_row(details, 0, "Dataset Path", self.dataset_path_var)
        self._add_status_row(details, 1, "Dataset Rows", self.dataset_rows_var)
        self._add_status_row(details, 2, "CSV Files In Node", self.dataset_file_count_var)
        self._add_status_row(details, 3, "Latest Validation", self.validation_status_var)
        self._add_status_row(details, 4, "Validation Report", self.validation_report_var)
        self._add_status_row(details, 5, "Latest Global Model", self.latest_global_model_var)
        self._add_status_row(details, 6, "Latest Local Update", self.latest_local_update_var)

    def _build_run_round_tab(self) -> None:
        ttk.Label(
            self.run_round_tab,
            text=(
                "Run the local node flow in three steps: validate the dataset, adapt the local model from the "
                "cached global checkpoint, or complete a full sync round where this node sends its update and "
                "then receives the refreshed global model for the next round."
            ),
            wraplength=920,
            justify="left",
        ).pack(anchor="w")

        controls = ttk.LabelFrame(self.run_round_tab, text="Round Controls", padding=12)
        controls.pack(fill="x", pady=(16, 0))
        validate_button = ttk.Button(controls, text="Validate Dataset", command=self.validate_selected_dataset)
        train_button = ttk.Button(controls, text="Train Local Node", command=self.train_selected_dataset)
        sync_button = ttk.Button(controls, text="Run Node Sync Round", command=self.sync_with_aggregator)
        validate_button.pack(side="left")
        train_button.pack(side="left", padx=(8, 0))
        sync_button.pack(side="left", padx=(8, 0))
        self.busy_widgets.extend([validate_button, train_button, sync_button])

        status_frame = ttk.LabelFrame(self.run_round_tab, text="Run Status", padding=12)
        status_frame.pack(fill="x", pady=(18, 0))
        self._add_status_row(status_frame, 0, "Dataset Validation", self.validation_status_var)
        self._add_status_row(status_frame, 1, "Validation Report", self.validation_report_var)
        self._add_status_row(status_frame, 2, "Local Adaptation", self.training_status_var)
        self._add_status_row(status_frame, 3, "Round Synchronization", self.sync_status_var)
        self._add_status_row(status_frame, 4, "Local Model Version", self.local_model_version_var)
        self._add_status_row(status_frame, 5, "Global Model Version", self.global_model_version_var)

        ttk.Label(
            self.run_round_tab,
            text=(
                "Use 'Train Local Node' for standalone local adaptation. "
                "Use 'Run Node Sync Round' when the aggregator is active. The node will train from its cached "
                "global model, send its local update, and then wait for the refreshed global model."
            ),
            wraplength=920,
            justify="left",
        ).pack(anchor="w", pady=(14, 0))

    def _build_results_tab(self) -> None:
        ttk.Label(
            self.results_tab,
            text=(
                "Inspect model metrics, transfer artifacts, and evaluation outputs. "
                "Batch evaluation scores the selected CSV using the current local model."
            ),
            wraplength=920,
            justify="left",
        ).pack(anchor="w")

        top = ttk.Frame(self.results_tab)
        top.pack(fill="x", pady=(16, 0))
        self._build_value_card(top, "Training Accuracy", self.training_accuracy_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._build_value_card(top, "Training Loss", self.training_loss_var).pack(side="left", fill="x", expand=True, padx=8)
        self._build_value_card(top, "Evaluation Output", self.evaluation_output_var).pack(side="left", fill="x", expand=True, padx=(8, 0))

        evaluation = ttk.LabelFrame(self.results_tab, text="Batch Evaluation", padding=12)
        evaluation.pack(fill="x", pady=(18, 0))
        ttk.Label(
            evaluation,
            text=(
                "Run the selected CSV against the current node model. "
                "The scored output CSV will be saved in the node reports folder."
            ),
            wraplength=920,
            justify="left",
        ).pack(anchor="w")
        ttk.Label(evaluation, text="Selected CSV", style="CardTitle.TLabel").pack(anchor="w", pady=(10, 0))
        ttk.Label(evaluation, textvariable=self.dataset_var, style="Value.TLabel").pack(anchor="w", pady=(4, 0))
        evaluate_button = ttk.Button(evaluation, text="Run Batch Evaluation", command=self.evaluate_selected_dataset)
        evaluate_button.pack(anchor="w", pady=(10, 0))
        ttk.Label(evaluation, textvariable=self.evaluation_status_var, style="Value.TLabel").pack(anchor="w", pady=(10, 0))
        self.busy_widgets.append(evaluate_button)

        logs = ttk.LabelFrame(self.results_tab, text="Logs", padding=12)
        logs.pack(fill="both", expand=True, pady=(18, 0))
        log_controls = ttk.Frame(logs)
        log_controls.pack(fill="x")
        ttk.Label(log_controls, text="Log Stream").pack(side="left")
        self.log_combo = ttk.Combobox(
            log_controls,
            textvariable=self.log_choice_var,
            state="readonly",
            values=list(self.LOG_LABEL_TO_KEY.keys()),
            width=18,
        )
        self.log_combo.pack(side="left", padx=(8, 0))
        refresh_button = ttk.Button(log_controls, text="Refresh Logs", command=self.refresh_logs)
        refresh_button.pack(side="left", padx=(8, 0))
        self.busy_widgets.extend([self.log_combo, refresh_button])

        self.logs_text = ScrolledText(logs, wrap="word", height=18, font=("Consolas", 10))
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

    def _run_background_task(self, task_name: str, task_callable: Callable[[], Any]) -> None:
        if self.is_busy:
            messagebox.showinfo("Busy", "Another action is already running. Please wait for it to finish.")
            return
        self._set_busy(True)
        self.current_status_var.set(f"{task_name} is running...")

        def _worker() -> None:
            try:
                self.background_results.put(("ok", task_name, task_callable()))
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
                    self._handle_failed_task(task_name, payload)
                self.refresh_node_status()
                self.refresh_logs()
        except Empty:
            pass
        self.after(150, self._process_background_results)

    def _handle_completed_task(self, task_name: str, payload: Any) -> None:
        self.current_status_var.set(f"{task_name} completed")
        if task_name == "Validate Dataset":
            self.validation_status_var.set(f"Validation: {payload['status']}")
            self.validation_report_var.set(self._format_path_label(payload["report_path"]))
            messagebox.showinfo("Dataset Validation Complete", f"Status: {payload['status']}\nReport:\n{payload['report_path']}")
            return
        if task_name == "Train Local Node":
            self.training_status_var.set(f"Local adaptation complete. Accuracy: {payload['validation_accuracy']:.4f}")
            self.validation_status_var.set(f"Validation: {payload['validation_result']['status']}")
            self.validation_report_var.set(self._format_path_label(payload["validation_report_path"]))
            messagebox.showinfo(
                "Local Adaptation Complete",
                f"Model saved to:\n{payload['model_path']}\n\nValidation accuracy: {payload['validation_accuracy']:.4f}",
            )
            return
        if task_name == "Run Full Sync Round":
            self.sync_status_var.set(f"Round complete: {payload['round_name']}")
            self.training_status_var.set(
                f"Local adaptation complete. Accuracy: {payload['training_result']['validation_accuracy']:.4f}"
            )
            self.validation_status_var.set(f"Validation: {payload['training_result']['validation_result']['status']}")
            self.validation_report_var.set(self._format_path_label(payload["training_result"]["validation_report_path"]))
            self.global_model_version_var.set(payload["round_name"])
            messagebox.showinfo(
                "Round Synchronization Complete",
                (
                    f"Completed {payload['round_name']} successfully.\n\n"
                    f"Refreshed global model:\n{payload['refreshed_global_model_path']}"
                ),
            )
            return
        if task_name == "Run Batch Evaluation":
            positive_predictions = sum(row["predicted_label"] for row in payload["predictions"])
            accuracy_text = "Accuracy: N/A" if payload["accuracy"] is None else f"Accuracy: {payload['accuracy']:.4f}"
            self.evaluation_status_var.set(
                f"Rows scored: {len(payload['predictions'])}, positive predictions: {positive_predictions}, {accuracy_text}"
            )
            self.evaluation_output_var.set(self._format_path_label(payload["prediction_output_path"]))
            messagebox.showinfo(
                "Batch Evaluation Complete",
                f"Rows scored: {len(payload['predictions'])}\n{accuracy_text}\n\nOutput:\n{payload['prediction_output_path']}",
            )

    def _handle_failed_task(self, task_name: str, error: Exception) -> None:
        self.current_status_var.set(f"{task_name} failed")
        if task_name == "Validate Dataset":
            self.validation_status_var.set("Validation: failed")
        if task_name == "Train Local Node":
            self.training_status_var.set("Local adaptation failed")
        if task_name == "Run Full Sync Round":
            self.sync_status_var.set("Round synchronization failed")
        if task_name == "Run Batch Evaluation":
            self.evaluation_status_var.set("Batch evaluation failed")
        messagebox.showerror(task_name, str(error))

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
        self._update_selected_dataset_metadata()

    def refresh_node_status(self) -> None:
        self.node_status = get_research_node_status(self.config_path)
        self.node_label_var.set(self.node_status["node_label"])
        self.hospital_id_var.set(self.node_status["hospital_name"])
        self.local_model_version_var.set(self.node_status["local_model_version"])
        self.global_model_version_var.set(self.node_status["current_global_version"])
        self.model_size_var.set(self.node_status["model_size_text"])
        self.latest_global_size_var.set(self.node_status["latest_global_model_size_text"])
        self.latest_update_size_var.set(self.node_status["latest_local_update_size_text"])
        self.latest_global_model_var.set(self._format_path_label(self.node_status["latest_global_model_path"]))
        self.latest_local_update_var.set(self._format_path_label(self.node_status["latest_local_update_path"]))
        self.training_accuracy_var.set(self._format_metric(self.node_status["training_accuracy"]))
        self.training_loss_var.set(self._format_metric(self.node_status["training_loss"]))
        self.dataset_path_var.set(str(self.node_status["dataset_path"]))
        self.dataset_rows_var.set(self._format_count(self.node_status["dataset_row_count"]))
        self.dataset_file_count_var.set(str(self.node_status["dataset_file_count"]))
        self.validation_status_var.set(self._format_validation_status(self.node_status["latest_validation_status"]))
        self.validation_report_var.set(self._format_path_label(self.node_status["latest_validation_report"]))
        if self.node_status["latest_evaluation_output_path"] is not None:
            self.evaluation_output_var.set(self._format_path_label(self.node_status["latest_evaluation_output_path"]))
        self.refresh_dataset_list()

    def refresh_logs(self) -> None:
        log_key = self.LOG_LABEL_TO_KEY.get(self.log_choice_var.get() or "training", "training")
        status = get_research_node_status(self.config_path)
        log_path = status["logs"].get(log_key)
        if log_path is None:
            return
        self.logs_text.delete("1.0", tk.END)
        self.logs_text.insert("1.0", read_recent_log_lines(log_path))

    def choose_dataset_file(self) -> None:
        selected_path = filedialog.askopenfilename(
            title="Choose Node Dataset CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not selected_path:
            return
        self.selected_upload_path = Path(selected_path)
        self.upload_path_var.set(f"Selected file: {self.selected_upload_path}")
        self.upload_status_var.set("Ready to import into node folder")

    def upload_selected_dataset(self) -> None:
        if self.selected_upload_path is None:
            messagebox.showwarning("No File Selected", "Choose a CSV file first.")
            return
        destination_path = copy_dataset_into_uploads(self.config_path, self.selected_upload_path)
        self.upload_status_var.set(f"File copied into node folder: {destination_path.name}")
        self.dataset_var.set(destination_path.name)
        self.current_status_var.set("Node dataset imported successfully")
        self.refresh_node_status()

    def _handle_dataset_selection(self, event: tk.Event[tk.Listbox]) -> None:
        selection = self.dataset_listbox.curselection()
        if selection:
            self.dataset_var.set(self.dataset_listbox.get(selection[0]))
            self._update_selected_dataset_metadata()

    def _handle_dataset_combo_selection(self, event: tk.Event[ttk.Combobox]) -> None:
        self._update_selected_dataset_metadata()

    def validate_selected_dataset(self) -> None:
        dataset_name = self.dataset_var.get().strip() or None
        self.validation_status_var.set("Validation is running...")
        self._run_background_task(
            "Validate Dataset",
            lambda: validate_training_dataset(config_path=self.config_path, dataset_filename=dataset_name),
        )

    def train_selected_dataset(self) -> None:
        dataset_name = self.dataset_var.get().strip() or None
        self.training_status_var.set("Local adaptation is running...")
        self._run_background_task(
            "Train Local Node",
            lambda: train_local_model(config_path=self.config_path, dataset_filename=dataset_name),
        )

    def sync_with_aggregator(self) -> None:
        dataset_name = self.dataset_var.get().strip() or None
        self.sync_status_var.set("Running local sync round from cached global model...")
        self._run_background_task(
            "Run Full Sync Round",
            lambda: run_hospital_federated_round(
                config_path=self.config_path,
                dataset_filename=dataset_name,
                receive_global_model_callable=self.receive_global_model_callable,
                send_local_update_callable=self.send_local_update_callable,
            ),
        )

    def evaluate_selected_dataset(self) -> None:
        dataset_name = self.dataset_var.get().strip()
        if not dataset_name:
            messagebox.showwarning("No CSV Selected", "Select a node CSV file first.")
            return
        _, paths = load_hospital_context(self.config_path)
        input_path = paths["uploads_dir"] / dataset_name
        self.evaluation_status_var.set("Batch evaluation is running...")
        self._run_background_task(
            "Run Batch Evaluation",
            lambda: predict_from_csv(config_path=self.config_path, input_path=input_path),
        )

    @staticmethod
    def _format_metric(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    @staticmethod
    def _format_count(value: int | None) -> str:
        return "Unknown" if value is None else str(value)

    @staticmethod
    def _format_validation_status(value: str | None) -> str:
        return "Validation: not run yet" if not value else f"Validation: {value}"

    @staticmethod
    def _format_path_label(path_value: Path | None) -> str:
        return "N/A" if path_value is None else path_value.name

    def _update_selected_dataset_metadata(self) -> None:
        dataset_name = self.dataset_var.get().strip()
        _, paths = load_hospital_context(self.config_path)
        if not dataset_name:
            self.dataset_path_var.set("N/A")
            self.dataset_rows_var.set("Unknown")
            return
        dataset_path = paths["uploads_dir"] / dataset_name
        self.dataset_path_var.set(str(dataset_path))
        if not dataset_path.exists():
            self.dataset_rows_var.set("Unknown")
            return
        try:
            self.dataset_rows_var.set(str(len(load_csv_records(dataset_path))))
        except Exception:
            self.dataset_rows_var.set("Unknown")


def launch_hospital_gui(
    config_path: Path,
    receive_global_model_callable: Callable[..., dict[str, Any]],
    send_local_update_callable: Callable[..., dict[str, Any]],
) -> None:
    """Launch the simplified research-demo node GUI."""
    app = ResearchNodeApp(
        config_path=config_path,
        receive_global_model_callable=receive_global_model_callable,
        send_local_update_callable=send_local_update_callable,
    )
    app.mainloop()
