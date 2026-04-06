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
    format_example_patient_values,
    get_example_patient_details,
    get_example_patient_values_for_hospital,
    get_feature_columns_for_hospital,
    get_hospital_dashboard_status,
    get_prediction_range_guide_for_hospital,
    list_available_dataset_files,
    read_recent_log_lines,
)
from FedLite_Project.Shared_Assets.common_utilities.hospital_quality_reports import (
    validate_training_dataset,
)
from FedLite_Project.Shared_Assets.common_utilities.local_ml_pipeline import (
    load_hospital_context,
    predict_from_csv,
    predict_from_patient_values,
    train_local_model,
)


class HospitalClientApp(tk.Tk):
    """Tkinter client that wraps the existing hospital backend modules."""

    PATIENT_DETAIL_FIELDS = [
        ("first_name", "First Name"),
        ("last_name", "Last Name"),
        ("gender", "Gender"),
        ("date_of_birth", "Date of Birth"),
        ("contact_number", "Contact Number"),
        ("department", "Department"),
        ("attending_doctor", "Attending Doctor"),
        ("address", "Address"),
        ("visit_notes", "Visit Notes"),
    ]

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
        self.example_patient_values = get_example_patient_values_for_hospital(self.config_path)
        self.example_patient_details = get_example_patient_details()
        self.prediction_range_guide = get_prediction_range_guide_for_hospital(self.config_path)

        self.title(f"FedLiteCare Hospital Client - {self.dashboard_status['hospital_name']}")
        self.geometry("1120x780")
        self.minsize(980, 700)

        self.background_results: Queue[tuple[str, str, Any]] = Queue()
        self.busy_widgets: list[ttk.Widget] = []
        self.selected_upload_path: Path | None = None
        self.is_busy = False

        self.current_status_var = tk.StringVar(value="Ready for patient screening")
        self.training_status_var = tk.StringVar(value="No local processing yet")
        self.sync_status_var = tk.StringVar(value="No server update yet")
        self.validation_status_var = tk.StringVar(value="Data quality not checked")
        self.upload_status_var = tk.StringVar(value="No hospital CSV selected")
        self.prediction_result_var = tk.StringVar(value="No screening result yet")
        self.confidence_var = tk.StringVar(value="Confidence: N/A")
        self.validation_report_var = tk.StringVar(value="Data check report: N/A")
        self.prediction_report_var = tk.StringVar(value="Patient report: N/A")
        self.prediction_registry_var = tk.StringVar(value="Predicted patients CSV: N/A")
        self.csv_prediction_status_var = tk.StringVar(value="CSV screening: not run yet")
        self.csv_prediction_output_var = tk.StringVar(value="Predicted CSV Output: N/A")
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
        self.patient_case_id_var = tk.StringVar(value="")
        self.patient_detail_vars = {
            field_name: tk.StringVar(value="")
            for field_name, _ in self.PATIENT_DETAIL_FIELDS
        }
        self.field_vars = {column: tk.StringVar(value="") for column in self.feature_columns}
        self.example_input_var = tk.StringVar(
            value="Example input: " + format_example_patient_values(self.example_patient_values)
        )
        self.range_guide_var = tk.StringVar(
            value="Allowed ranges:\n" + self.prediction_range_guide
        )

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
        ttk.Label(header, text="FedLiteCare Hospital Client", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Simple screening and daily update workspace",
            style="Subheader.TLabel",
        ).pack(anchor="w")

        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True)

        self.dashboard_tab = ttk.Frame(notebook, padding=12)
        self.upload_tab = ttk.Frame(notebook, padding=12)
        self.train_tab = ttk.Frame(notebook, padding=12)
        self.predict_tab_container, self.predict_tab = self._create_scrollable_tab(notebook, padding=12)
        self.sync_tab = ttk.Frame(notebook, padding=12)
        self.logs_tab = ttk.Frame(notebook, padding=12)

        notebook.add(self.dashboard_tab, text="Home")
        notebook.add(self.upload_tab, text="Patient Files")
        notebook.add(self.train_tab, text="Daily Processing")
        notebook.add(self.predict_tab_container, text="Patient Screening")
        notebook.add(self.sync_tab, text="Server Update")
        notebook.add(self.logs_tab, text="Activity")

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
        self._build_value_card(top_row, "Selected File", textvariable=self.dataset_var).pack(
            side="left", fill="x", expand=True, padx=8
        )
        self._build_value_card(top_row, "Local Screening Version", textvariable=self.model_version_var).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )

        middle_row = ttk.Frame(self.dashboard_tab)
        middle_row.pack(fill="x", pady=(12, 0))
        self._build_value_card(middle_row, "Server Version", textvariable=self.global_version_var).pack(
            side="left", fill="x", expand=True, padx=(0, 8)
        )
        self._build_value_card(middle_row, "Data Quality", textvariable=self.validation_status_var).pack(
            side="left", fill="x", expand=True, padx=8
        )
        self._build_value_card(middle_row, "Server Update", textvariable=self.sync_status_var).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )

        status_frame = ttk.LabelFrame(self.dashboard_tab, text="Hospital Status", padding=12)
        status_frame.pack(fill="x", pady=(16, 0))
        self._add_status_row(status_frame, 0, "Current Status", self.current_status_var)
        self._add_status_row(status_frame, 1, "Data Quality", self.validation_status_var)
        self._add_status_row(status_frame, 2, "Local Processing", self.training_status_var)
        self._add_status_row(status_frame, 3, "Server Update", self.sync_status_var)
        self._add_status_row(status_frame, 4, "Latest Data Check", self.validation_report_var)
        self._add_status_row(status_frame, 5, "Latest Patient Report", self.prediction_report_var)

        refresh_button = ttk.Button(
            self.dashboard_tab,
            text="Refresh Status",
            command=self.refresh_dashboard,
        )
        refresh_button.pack(anchor="e", pady=(12, 0))
        self.busy_widgets.append(refresh_button)

    def _build_upload_tab(self) -> None:
        info = ttk.Label(
            self.upload_tab,
            text=(
                "Bring hospital CSV files into this folder. These files can then be used "
                "for daily processing or patient screening."
            ),
            wraplength=820,
            justify="left",
        )
        info.pack(anchor="w")

        controls = ttk.Frame(self.upload_tab)
        controls.pack(fill="x", pady=(14, 0))
        choose_button = ttk.Button(controls, text="Browse CSV File", command=self.choose_dataset_file)
        choose_button.pack(side="left")
        self.busy_widgets.append(choose_button)
        upload_button = ttk.Button(controls, text="Save File To Hospital", command=self.upload_selected_dataset)
        upload_button.pack(side="left", padx=(8, 0))
        self.busy_widgets.append(upload_button)

        ttk.Label(self.upload_tab, textvariable=self.upload_path_var).pack(anchor="w", pady=(14, 0))
        ttk.Label(self.upload_tab, textvariable=self.upload_status_var).pack(anchor="w", pady=(8, 0))

        dataset_frame = ttk.LabelFrame(self.upload_tab, text="Available Hospital CSV Files", padding=12)
        dataset_frame.pack(fill="both", expand=True, pady=(18, 0))
        self.dataset_listbox = tk.Listbox(dataset_frame, height=12)
        self.dataset_listbox.pack(fill="both", expand=True)
        self.dataset_listbox.bind("<<ListboxSelect>>", self._handle_dataset_selection)

    def _build_train_tab(self) -> None:
        form = ttk.Frame(self.train_tab)
        form.pack(fill="x")
        ttk.Label(form, text="Selected Hospital CSV").grid(row=0, column=0, sticky="w")
        self.dataset_combo = ttk.Combobox(form, textvariable=self.dataset_var, state="readonly", width=40)
        self.dataset_combo.grid(row=1, column=0, sticky="w", pady=(4, 0))
        refresh_button = ttk.Button(form, text="Refresh File List", command=self.refresh_dataset_list)
        refresh_button.grid(row=1, column=1, sticky="w", padx=(10, 0))
        self.busy_widgets.extend([self.dataset_combo, refresh_button])

        actions = ttk.Frame(self.train_tab)
        actions.pack(anchor="w", pady=(16, 0))
        validate_button = ttk.Button(actions, text="Check Data Quality", command=self.validate_selected_dataset)
        validate_button.pack(side="left")
        train_button = ttk.Button(actions, text="Process Local Data", command=self.train_selected_dataset)
        train_button.pack(side="left", padx=(8, 0))
        train_and_sync_button = ttk.Button(
            actions,
            text="Complete Daily Update",
            command=self.train_and_sync_selected_dataset,
        )
        train_and_sync_button.pack(side="left", padx=(8, 0))
        self.busy_widgets.extend([validate_button, train_button, train_and_sync_button])
        ttk.Label(self.train_tab, textvariable=self.training_status_var).pack(anchor="w", pady=(10, 0))
        ttk.Label(self.train_tab, textvariable=self.validation_status_var).pack(anchor="w", pady=(6, 0))
        ttk.Label(self.train_tab, textvariable=self.validation_report_var).pack(anchor="w", pady=(6, 0))
        ttk.Label(
            self.train_tab,
            text=(
                "Use 'Check Data Quality' to review a CSV before processing. "
                "Use 'Process Local Data' to refresh the hospital model locally. "
                "Use 'Complete Daily Update' when the central server is running."
            ),
            wraplength=840,
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

    def _build_predict_tab(self) -> None:
        info = ttk.Label(
            self.predict_tab,
            text=(
                "Use this screen to check one patient or a whole CSV file. "
                "The app saves readable outputs automatically for hospital review."
            ),
            wraplength=840,
            justify="left",
        )
        info.pack(anchor="w")

        example_frame = ttk.LabelFrame(self.predict_tab, text="Quick Example", padding=12)
        example_frame.pack(fill="x", pady=(12, 0))
        ttk.Label(
            example_frame,
            textvariable=self.example_input_var,
            wraplength=840,
            justify="left",
        ).pack(anchor="w")

        range_frame = ttk.LabelFrame(self.predict_tab, text="Field Guide", padding=12)
        range_frame.pack(fill="x", pady=(12, 0))
        ttk.Label(
            range_frame,
            textvariable=self.range_guide_var,
            wraplength=840,
            justify="left",
        ).pack(anchor="w")

        csv_prediction_frame = ttk.LabelFrame(self.predict_tab, text="Predict From Selected CSV", padding=12)
        csv_prediction_frame.pack(fill="x", pady=(12, 0))
        ttk.Label(
            csv_prediction_frame,
            text=(
                "Use the selected CSV from the Patient Files or Daily Processing sections. "
                "A scored CSV copy will be saved into the hospital prediction reports folder."
            ),
            wraplength=840,
            justify="left",
        ).pack(anchor="w")
        ttk.Label(
            csv_prediction_frame,
            text="Selected CSV",
            style="CardTitle.TLabel",
        ).pack(anchor="w", pady=(10, 0))
        ttk.Label(
            csv_prediction_frame,
            textvariable=self.dataset_var,
            style="Value.TLabel",
        ).pack(anchor="w", pady=(4, 0))
        predict_csv_button = ttk.Button(
            csv_prediction_frame,
            text="Screen Selected CSV",
            command=self.predict_selected_csv,
        )
        predict_csv_button.pack(anchor="w", pady=(10, 0))
        self.busy_widgets.append(predict_csv_button)
        ttk.Label(
            csv_prediction_frame,
            textvariable=self.csv_prediction_status_var,
            style="Value.TLabel",
        ).pack(anchor="w", pady=(10, 0))
        ttk.Label(
            csv_prediction_frame,
            textvariable=self.csv_prediction_output_var,
            style="Value.TLabel",
        ).pack(anchor="w", pady=(8, 0))

        patient_frame = ttk.LabelFrame(self.predict_tab, text="Patient Details", padding=12)
        patient_frame.pack(fill="x", pady=(12, 0))

        ttk.Label(patient_frame, text="Patient Case ID").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(patient_frame, textvariable=self.patient_case_id_var, width=22).grid(
            row=0,
            column=1,
            sticky="w",
            pady=6,
        )
        ttk.Label(
            patient_frame,
            text="Leave blank to auto-generate a case ID.",
        ).grid(row=0, column=2, columnspan=2, sticky="w", padx=(12, 0), pady=6)

        for index, (field_name, label_text) in enumerate(self.PATIENT_DETAIL_FIELDS, start=1):
            row = ((index - 1) // 2) + 1
            column_position = ((index - 1) % 2) * 2
            ttk.Label(patient_frame, text=label_text).grid(
                row=row,
                column=column_position,
                sticky="w",
                padx=(0, 8),
                pady=6,
            )
            ttk.Entry(patient_frame, textvariable=self.patient_detail_vars[field_name], width=30).grid(
                row=row,
                column=column_position + 1,
                sticky="w",
                pady=6,
            )
        ttk.Label(
            patient_frame,
            text="Patient identity details stay local to the hospital and are not used as training labels.",
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(10, 0))

        form = ttk.LabelFrame(self.predict_tab, text="Diabetes Screening Inputs", padding=12)
        form.pack(fill="x", pady=(12, 0))

        for index, column in enumerate(self.feature_columns):
            row = (index // 2)
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
        predict_button = ttk.Button(actions, text="Screen Patient Risk", command=self.predict_patient_risk)
        predict_button.pack(side="left")
        example_button = ttk.Button(actions, text="Load Sample Input", command=self.load_example_patient_values)
        example_button.pack(side="left", padx=(8, 0))
        clear_button = ttk.Button(actions, text="Clear Form", command=self.clear_prediction_fields)
        clear_button.pack(side="left", padx=(8, 0))
        self.busy_widgets.extend([predict_button, example_button, clear_button])

        result_frame = ttk.LabelFrame(self.predict_tab, text="Screening Result", padding=12)
        result_frame.pack(fill="x", pady=(16, 0))
        ttk.Label(result_frame, textvariable=self.prediction_result_var, style="Value.TLabel").pack(anchor="w")
        ttk.Label(result_frame, textvariable=self.confidence_var, style="Value.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Label(result_frame, textvariable=self.prediction_report_var, style="Value.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Label(result_frame, textvariable=self.prediction_registry_var, style="Value.TLabel").pack(anchor="w", pady=(8, 0))

    def _build_sync_tab(self) -> None:
        info = ttk.Label(
            self.sync_tab,
            text=(
                "Use this screen when the server is already running. "
                "The hospital client will receive the latest server model, process local data, and send a daily update."
            ),
            wraplength=840,
            justify="left",
        )
        info.pack(anchor="w")

        sync_button = ttk.Button(self.sync_tab, text="Send Daily Update", command=self.sync_with_aggregator)
        sync_button.pack(anchor="w", pady=(16, 0))
        self.busy_widgets.append(sync_button)
        ttk.Label(self.sync_tab, textvariable=self.sync_status_var).pack(anchor="w", pady=(10, 0))

    def _build_logs_tab(self) -> None:
        controls = ttk.Frame(self.logs_tab)
        controls.pack(fill="x")
        ttk.Label(controls, text="Activity Log").pack(side="left")
        self.log_combo = ttk.Combobox(
            controls,
            textvariable=self.log_choice_var,
            state="readonly",
            values=["training", "prediction", "sync", "transfer"],
            width=18,
        )
        self.log_combo.pack(side="left", padx=(8, 0))
        self.busy_widgets.append(self.log_combo)
        refresh_button = ttk.Button(controls, text="Refresh Activity", command=self.refresh_logs)
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

    def _create_scrollable_tab(
        self,
        notebook: ttk.Notebook,
        padding: int,
    ) -> tuple[ttk.Frame, ttk.Frame]:
        outer_frame = ttk.Frame(notebook)
        canvas = tk.Canvas(outer_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
        content_frame = ttk.Frame(canvas, padding=padding)
        content_window = canvas.create_window((0, 0), window=content_frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _update_scroll_region(_event: tk.Event[tk.Misc] | None = None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _match_content_width(event: tk.Event[tk.Misc]) -> None:
            canvas.itemconfigure(content_window, width=event.width)

        def _on_mousewheel(event: tk.Event[tk.Misc]) -> None:
            if content_frame.winfo_reqheight() <= canvas.winfo_height():
                return
            delta = int(-event.delta / 120) if getattr(event, "delta", 0) else 0
            if delta:
                canvas.yview_scroll(delta, "units")

        def _bind_mousewheel(_event: tk.Event[tk.Misc]) -> None:
            self.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(_event: tk.Event[tk.Misc]) -> None:
            self.unbind_all("<MouseWheel>")

        content_frame.bind("<Configure>", _update_scroll_region)
        canvas.bind("<Configure>", _match_content_width)
        outer_frame.bind("<Enter>", _bind_mousewheel)
        outer_frame.bind("<Leave>", _unbind_mousewheel)

        return outer_frame, content_frame

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
                    if task_name == "Process Local Data":
                        self.training_status_var.set("Local processing failed")
                    if task_name == "Check Data Quality":
                        self.validation_status_var.set("Data quality check failed")
                    if task_name == "Screen Patient Risk":
                        self.prediction_result_var.set("No screening result yet")
                        self.confidence_var.set("Confidence: N/A")
                    if task_name == "Screen Selected CSV":
                        self.csv_prediction_status_var.set("CSV screening failed")
                        self.csv_prediction_output_var.set("Predicted CSV Output: N/A")
                    if task_name == "Send Daily Update":
                        self.sync_status_var.set("Server update failed")
                    messagebox.showerror(task_name, str(payload))
                self.refresh_dashboard()
                self.refresh_logs()
        except Empty:
            pass

        self.after(150, self._process_background_results)

    def _handle_completed_task(self, task_name: str, payload: Any) -> None:
        self.current_status_var.set(f"{task_name} completed")

        if task_name == "Process Local Data":
            self.training_status_var.set(
                f"Local processing complete. Accuracy: {payload['validation_accuracy']:.4f}"
            )
            self.validation_status_var.set(f"Data quality: {payload['validation_result']['status']}")
            self.validation_report_var.set(f"Data check report: {payload['validation_report_path'].name}")
            messagebox.showinfo(
                "Local Processing Complete",
                (
                    f"Model saved to:\n{payload['model_path']}\n\n"
                    f"Data quality: {payload['validation_result']['status']}\n"
                    f"Data check report: {payload['validation_report_path']}\n"
                    f"Validation accuracy: {payload['validation_accuracy']:.4f}"
                ),
            )
            return

        if task_name == "Check Data Quality":
            self.validation_status_var.set(f"Data quality: {payload['status']}")
            self.validation_report_var.set(f"Data check report: {payload['report_path'].name}")
            self.current_status_var.set("Data quality check completed")
            messagebox.showinfo(
                "Data Quality Complete",
                f"Status: {payload['status']}\nReport saved to:\n{payload['report_path']}",
            )
            return

        if task_name == "Screen Patient Risk":
            self.prediction_result_var.set(f"Screening Result: {payload['result_label']}")
            self.confidence_var.set(f"Confidence Score: {payload['confidence_score']:.4f}")
            self.prediction_report_var.set(f"Patient report: {payload['report_path'].name}")
            self.prediction_registry_var.set(f"Predicted patients CSV: {payload['predicted_patients_path'].name}")
            self.patient_case_id_var.set(payload["patient_case_id"])
            self.current_status_var.set("Patient screening saved to predicted patients CSV")
            return

        if task_name == "Screen Selected CSV":
            positive_predictions = sum(
                row["predicted_label"] for row in payload["predictions"]
            )
            accuracy_text = (
                "Accuracy: N/A"
                if payload["accuracy"] is None
                else f"Accuracy: {payload['accuracy']:.4f}"
            )
            self.csv_prediction_status_var.set(
                (
                    f"CSV screening complete. Rows: {len(payload['predictions'])}, "
                    f"High-risk predictions: {positive_predictions}, {accuracy_text}"
                )
            )
            self.csv_prediction_output_var.set(
                f"Predicted CSV Output: {payload['prediction_output_path'].name}"
            )
            self.current_status_var.set("CSV screening output saved")
            messagebox.showinfo(
                "CSV Screening Complete",
                (
                    f"Scored file:\n{payload['input_path']}\n\n"
                    f"Rows scored: {len(payload['predictions'])}\n"
                    f"{accuracy_text}\n"
                    f"Output saved to:\n{payload['prediction_output_path']}"
                ),
            )
            return

        if task_name == "Send Daily Update":
            self.sync_status_var.set(f"Daily update complete for {payload['round_name']}")
            self.validation_status_var.set(
                f"Data quality: {payload['training_result']['validation_result']['status']}"
            )
            self.validation_report_var.set(
                f"Data check report: {payload['training_result']['validation_report_path'].name}"
            )
            messagebox.showinfo(
                "Daily Update Complete",
                (
                    f"Completed {payload['round_name']}.\n"
                    f"Data quality: {payload['training_result']['validation_result']['status']}\n"
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
            self.validation_status_var.set(f"Data quality: {latest_validation_status}")
        self.validation_report_var.set(
            "Data check report: N/A" if latest_validation_report is None else f"Data check report: {latest_validation_report.name}"
        )
        self.prediction_report_var.set(
            "Patient report: N/A" if latest_prediction_report is None else f"Patient report: {latest_prediction_report.name}"
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
            title="Choose Hospital CSV",
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
        self.upload_status_var.set(f"File saved to hospital folder: {destination_path.name}")
        self.dataset_var.set(destination_path.name)
        self.current_status_var.set("Hospital file saved successfully")
        self.refresh_dashboard()

    def _handle_dataset_selection(self, event: tk.Event[tk.Listbox]) -> None:
        selection = self.dataset_listbox.curselection()
        if not selection:
            return
        self.dataset_var.set(self.dataset_listbox.get(selection[0]))

    def train_selected_dataset(self) -> None:
        dataset_name = self.dataset_var.get().strip() or None
        self.training_status_var.set("Local processing is running...")
        self._run_background_task(
            "Process Local Data",
            lambda: train_local_model(
                config_path=self.config_path,
                dataset_filename=dataset_name,
            ),
        )

    def validate_selected_dataset(self) -> None:
        dataset_name = self.dataset_var.get().strip() or None
        self.validation_status_var.set("Data quality check is running...")
        self._run_background_task(
            "Check Data Quality",
            lambda: validate_training_dataset(
                config_path=self.config_path,
                dataset_filename=dataset_name,
            ),
        )

    def train_and_sync_selected_dataset(self) -> None:
        self.training_status_var.set("Waiting for server, then processing and sending daily update...")
        self.sync_with_aggregator()

    def clear_prediction_fields(self) -> None:
        for variable in self.field_vars.values():
            variable.set("")
        for variable in self.patient_detail_vars.values():
            variable.set("")
        self.prediction_result_var.set("No screening result yet")
        self.confidence_var.set("Confidence: N/A")
        self.prediction_report_var.set("Patient report: N/A")
        self.prediction_registry_var.set("Predicted patients CSV: N/A")
        self.patient_case_id_var.set("")
        self.current_status_var.set("Patient form cleared")

    def load_example_patient_values(self) -> None:
        for column, variable in self.field_vars.items():
            variable.set(self.example_patient_values.get(column, "0"))
        for field_name, variable in self.patient_detail_vars.items():
            variable.set(self.example_patient_details.get(field_name, ""))
        self.current_status_var.set("Sample patient values loaded")

    def predict_patient_risk(self) -> None:
        patient_values = {
            column: variable.get().strip()
            for column, variable in self.field_vars.items()
        }
        patient_metadata = {
            field_name: variable.get().strip()
            for field_name, variable in self.patient_detail_vars.items()
        }
        self.prediction_result_var.set("Running screening...")
        self.confidence_var.set("Confidence: calculating...")
        self._run_background_task(
            "Screen Patient Risk",
            lambda: predict_from_patient_values(
                config_path=self.config_path,
                patient_values=patient_values,
                patient_metadata={
                    "patient_case_id": self.patient_case_id_var.get().strip(),
                    **patient_metadata,
                },
            ),
        )

    def predict_selected_csv(self) -> None:
        dataset_name = self.dataset_var.get().strip()
        if not dataset_name:
            messagebox.showwarning("No CSV Selected", "Select a hospital CSV file first.")
            return

        _, paths = load_hospital_context(self.config_path)
        input_path = paths["uploads_dir"] / dataset_name
        self.csv_prediction_status_var.set("CSV screening is running...")
        self.csv_prediction_output_var.set("Predicted CSV Output: generating...")
        self._run_background_task(
            "Screen Selected CSV",
            lambda: predict_from_csv(
                config_path=self.config_path,
                input_path=input_path,
            ),
        )

    def sync_with_aggregator(self) -> None:
        dataset_name = self.dataset_var.get().strip() or None
        self.sync_status_var.set("Waiting for server and running daily update...")
        self._run_background_task(
            "Send Daily Update",
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
