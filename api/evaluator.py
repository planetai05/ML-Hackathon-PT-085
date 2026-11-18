"""Background evaluation service for evaluator_test.zip (Laptop 1).

Manages background evaluation jobs:
- Inspects data/uploads/evaluator_test.zip
- Unpacks to temporary scratch folder (leaving original ZIP untouched)
- Evaluates all 10 trips using the production XGBoost + HMM/Viterbi pipeline
- Maintains strict ground-truth isolation (evaluates .route strictly post-prediction)
- Calculates aggregate metrics and writes CSV to results/evaluator_test/
- Provides thread-safe job status polling, results retrieval, and CSV download
"""
from __future__ import annotations

import csv
import datetime
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from typing import Any, Dict, List, Optional

log = logging.getLogger("pathsnap.evaluator")

# Cached device string
_DEVICE_CACHE: Optional[str] = None


def detect_execution_device() -> str:
    """Detect whether real GPU acceleration is active or CPU multithreading fallback is used."""
    global _DEVICE_CACHE
    if _DEVICE_CACHE is not None:
        return _DEVICE_CACHE

    gpu_name = "NVIDIA GPU"
    gpu_detected = False
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            gpu_detected = True
            gpu_name = res.stdout.strip().split("\n")[0].strip()
    except Exception:
        pass

    cuda_working = False
    try:
        import json
        import numpy as np
        import xgboost as xgb

        c = xgb.XGBClassifier(device="cuda", n_estimators=1)
        c.fit(np.zeros((10, 2)), np.array([0, 1] * 5))
        cfg = json.loads(c.get_booster().save_config())
        internal_dev = cfg.get("learner", {}).get("generic_param", {}).get("device", "")
        if "cuda" in internal_dev.lower():
            cuda_working = True
    except Exception:
        cuda_working = False

    if cuda_working:
        _DEVICE_CACHE = f"GPU ({gpu_name})"
    elif gpu_detected:
        _DEVICE_CACHE = f"CPU (12-thread fallback; {gpu_name} detected, CUDA runtime inactive in Python)"
    else:
        _DEVICE_CACHE = "CPU"
    return _DEVICE_CACHE


def inspect_evaluator_zip(zip_path: str) -> Dict[str, Any]:
    """Check existence of the evaluator zip and enumerate contained trip IDs."""
    if not os.path.isfile(zip_path):
        return {
            "dataset_name": os.path.basename(zip_path),
            "zip_path": os.path.abspath(zip_path),
            "exists": False,
            "total_trips": 0,
            "trips": [],
            "execution_device": detect_execution_device(),
        }

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            names = z.namelist()
            # Trips are subdirectories containing .track files
            trips = set()
            for name in names:
                parts = name.replace("\\", "/").strip("/").split("/")
                if len(parts) >= 2 and parts[-1].endswith(".track"):
                    trips.add(parts[-2])
            trips_list = sorted(trips)
            return {
                "dataset_name": os.path.basename(zip_path),
                "zip_path": os.path.abspath(zip_path),
                "exists": True,
                "total_trips": len(trips_list),
                "trips": trips_list,
                "execution_device": detect_execution_device(),
            }
    except Exception as e:
        log.warning("error inspecting zip %s: %s", zip_path, e)
        return {
            "dataset_name": os.path.basename(zip_path),
            "zip_path": os.path.abspath(zip_path),
            "exists": False,
            "total_trips": 0,
            "trips": [],
            "execution_device": detect_execution_device(),
            "error": str(e),
        }


class EvaluatorJobManager:
    """Thread-safe background job manager for evaluator_test.zip runs."""

    def __init__(self, results_dir: str = "results/evaluator_test"):
        self.results_dir = results_dir
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.active_job_id: Optional[str] = None
        self.latest_job_id: Optional[str] = None
        self._lock = threading.Lock()
        os.makedirs(self.results_dir, exist_ok=True)

    def start_job(self, zip_path: str, predictor) -> str:
        with self._lock:
            # Check if an existing job is currently running
            if self.active_job_id and self.active_job_id in self.jobs:
                active_job = self.jobs[self.active_job_id]
                if active_job["status"] in ("queued", "running"):
                    raise RuntimeError(f"Job '{self.active_job_id}' is already running")

            if not os.path.isfile(zip_path):
                raise FileNotFoundError(f"Evaluator zip not found at {os.path.abspath(zip_path)}")

            job_id = uuid.uuid4().hex[:8]
            job_state = {
                "job_id": job_id,
                "status": "queued",
                "current_trip": None,
                "completed_trips": 0,
                "total_trips": 0,
                "progress_pct": 0.0,
                "results": None,
                "csv_path": None,
                "error": None,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
            self.jobs[job_id] = job_state
            self.active_job_id = job_id
            self.latest_job_id = job_id

        # Launch background thread
        thread = threading.Thread(
            target=self._run_job_thread,
            args=(job_id, zip_path, predictor),
            daemon=True,
            name=f"eval-job-{job_id}",
        )
        thread.start()
        log.info("started evaluation job %s for %s", job_id, zip_path)
        return job_id

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            return {
                "job_id": job["job_id"],
                "status": job["status"],
                "current_trip": job["current_trip"],
                "completed_trips": job["completed_trips"],
                "total_trips": job["total_trips"],
                "progress_pct": job["progress_pct"],
                "error": job["error"],
            }

    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            return job.get("results")

    def get_job_csv_path(self, job_id: str) -> Optional[str]:
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            return job.get("csv_path")

    def _run_job_thread(self, job_id: str, zip_path: str, predictor):
        t_start = time.perf_counter()
        temp_dir = None
        device = detect_execution_device()

        try:
            with self._lock:
                self.jobs[job_id]["status"] = "running"
                self.jobs[job_id]["updated_at"] = time.time()

            # 1. Inspect and extract zip into isolated scratch dir
            temp_dir = tempfile.mkdtemp(prefix=f"pathsnap_eval_{job_id}_")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(temp_dir)

            # Locate trip folders
            trip_dirs = []
            for root, dirs, files in os.walk(temp_dir):
                for d in dirs:
                    cand = os.path.join(root, d)
                    if os.path.isfile(os.path.join(cand, f"{d}.track")):
                        trip_dirs.append(cand)
            trip_dirs = sorted(trip_dirs, key=lambda p: os.path.basename(p))

            total_trips = len(trip_dirs)
            if total_trips == 0:
                raise ValueError("No trip folders containing .track files found in the ZIP archive")

            with self._lock:
                self.jobs[job_id]["total_trips"] = total_trips
                self.jobs[job_id]["updated_at"] = time.time()

            per_trip_results: List[Dict[str, Any]] = []

            # 2. Sequential evaluation over all trips
            for idx, trip_path in enumerate(trip_dirs):
                tid = os.path.basename(trip_path)
                with self._lock:
                    self.jobs[job_id]["current_trip"] = tid
                    self.jobs[job_id]["completed_trips"] = idx
                    self.jobs[job_id]["progress_pct"] = round((idx / total_trips) * 100, 1)
                    self.jobs[job_id]["updated_at"] = time.time()

                t_trip0 = time.perf_counter()
                try:
                    # Ground-truth isolation: predict_trip reads strictly .track, .nodes, .arcs
                    pred = predictor.predict_trip(trip_path)
                    # Ground truth scored strictly post-prediction
                    score = predictor.score(pred, trip_path)
                    dt_ms = round((time.perf_counter() - t_trip0) * 1000, 1)

                    # Subsample GPS points if > 1200 for responsive UI map rendering
                    raw_pts = pred.get("gps_points", [])
                    step = max(1, len(raw_pts) // 1200)
                    sampled_pts = raw_pts[::step]

                    trip_res = {
                        "trip_id": tid,
                        "n_points": pred.get("n_points", len(raw_pts)),
                        "classifier_only_top1": score.get("classifier_only_top1"),
                        "viterbi_top1": score.get("top1_accuracy"),
                        "route_continuity": score.get("route_continuity"),
                        "route_precision": score.get("route_precision"),
                        "route_recall": score.get("route_recall"),
                        "route_f1": score.get("route_f1"),
                        "latency_ms": dt_ms,
                        "status": "ok",
                        "error": None,
                        "gps_points": sampled_pts,
                        "predicted_route_geojson": pred.get("predicted_route_geojson"),
                    }
                except Exception as e:
                    dt_ms = round((time.perf_counter() - t_trip0) * 1000, 1)
                    log.exception("evaluation failed on trip %s: %s", tid, e)
                    trip_res = {
                        "trip_id": tid,
                        "n_points": 0,
                        "classifier_only_top1": None,
                        "viterbi_top1": None,
                        "route_continuity": None,
                        "route_precision": None,
                        "route_recall": None,
                        "route_f1": None,
                        "latency_ms": dt_ms,
                        "status": "failed",
                        "error": str(e),
                        "gps_points": [],
                        "predicted_route_geojson": None,
                    }
                per_trip_results.append(trip_res)

            # 3. Compute aggregate metrics
            total_duration_s = round(time.perf_counter() - t_start, 2)
            successful = [r for r in per_trip_results if r["status"] == "ok"]
            failed_count = len(per_trip_results) - len(successful)

            mean_clf_top1 = (
                round(sum(r["classifier_only_top1"] for r in successful) / len(successful), 4)
                if successful
                else None
            )
            mean_vit_top1 = (
                round(sum(r["viterbi_top1"] for r in successful) / len(successful), 4)
                if successful
                else None
            )
            mean_cont = (
                round(sum(r["route_continuity"] for r in successful) / len(successful), 4)
                if successful
                else None
            )
            mean_f1 = (
                round(sum(r["route_f1"] for r in successful) / len(successful), 4)
                if successful
                else None
            )
            mean_lat = (
                round(sum(r["latency_ms"] for r in successful) / len(successful), 1)
                if successful
                else None
            )

            # 4. Generate CSV report
            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"evaluator_test_results_{ts_str}_{job_id}.csv"
            csv_path = os.path.join(self.results_dir, csv_filename)

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "trip_id",
                    "n_points",
                    "classifier_only_top1",
                    "viterbi_top1",
                    "route_continuity",
                    "route_precision",
                    "route_recall",
                    "route_f1",
                    "latency_ms",
                    "status",
                    "error",
                ])
                for r in per_trip_results:
                    writer.writerow([
                        r["trip_id"],
                        r["n_points"],
                        r["classifier_only_top1"],
                        r["viterbi_top1"],
                        r["route_continuity"],
                        r["route_precision"],
                        r["route_recall"],
                        r["route_f1"],
                        r["latency_ms"],
                        r["status"],
                        r["error"] or "",
                    ])
                if successful:
                    writer.writerow([
                        "MEAN",
                        int(sum(r["n_points"] for r in successful) / len(successful)),
                        mean_clf_top1,
                        mean_vit_top1,
                        mean_cont,
                        round(sum(r["route_precision"] for r in successful) / len(successful), 4),
                        round(sum(r["route_recall"] for r in successful) / len(successful), 4),
                        mean_f1,
                        mean_lat,
                        "ok",
                        f"Evaluated {len(successful)}/{total_trips} trips on {device}",
                    ])

            results_payload = {
                "job_id": job_id,
                "status": "completed",
                "total_trips": total_trips,
                "successful_trips": len(successful),
                "failed_trips": failed_count,
                "mean_classifier_top1": mean_clf_top1,
                "mean_viterbi_top1": mean_vit_top1,
                "mean_route_continuity": mean_cont,
                "mean_route_f1": mean_f1,
                "mean_latency_ms": mean_lat,
                "total_evaluation_time_s": total_duration_s,
                "execution_device": device,
                "csv_path": csv_path,
                "per_trip": per_trip_results,
                "error": None,
            }

            with self._lock:
                self.jobs[job_id]["status"] = "completed"
                self.jobs[job_id]["completed_trips"] = total_trips
                self.jobs[job_id]["progress_pct"] = 100.0
                self.jobs[job_id]["current_trip"] = None
                self.jobs[job_id]["results"] = results_payload
                self.jobs[job_id]["csv_path"] = csv_path
                self.jobs[job_id]["updated_at"] = time.time()
                if self.active_job_id == job_id:
                    self.active_job_id = None

            log.info(
                "job %s completed: %d/%d trips ok in %.1fs (Mean Viterbi Top-1: %s, Continuity: %s) -> %s",
                job_id,
                len(successful),
                total_trips,
                total_duration_s,
                mean_vit_top1,
                mean_cont,
                csv_path,
            )

        except Exception as e:
            log.exception("fatal error in evaluation job %s: %s", job_id, e)
            with self._lock:
                self.jobs[job_id]["status"] = "failed"
                self.jobs[job_id]["error"] = str(e)
                self.jobs[job_id]["results"] = {
                    "job_id": job_id,
                    "status": "failed",
                    "total_trips": 0,
                    "successful_trips": 0,
                    "failed_trips": 0,
                    "execution_device": device,
                    "total_evaluation_time_s": round(time.perf_counter() - t_start, 2),
                    "error": str(e),
                    "per_trip": [],
                }
                self.jobs[job_id]["updated_at"] = time.time()
                if self.active_job_id == job_id:
                    self.active_job_id = None
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
