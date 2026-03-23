#!/usr/bin/env python3
"""
scripts/run_demo.py

Full end-to-end demo of the NeuroRehab system.
Works with simulated hardware — no EMG sensor or camera required.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --game flower_bloom --pattern good --duration 60
    python scripts/run_demo.py --pipeline-only  # No game window, just pipeline test
"""

import sys
import os
import time
import argparse
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_PATIENT_ID, EMG_WINDOW_SIZE, PATIENT_ID, MQTT_BROKER, MQTT_PORT
from data_acquisition.emg_reader import create_emg_reader
from data_acquisition.preprocessor import EMGPreprocessor
from data_acquisition.session_logger import SessionLogger
from vision.tracker import MovementTracker, get_finger_extensions, get_finger_moving
from ml.trainer import PerformancePredictor, ProgressTracker, PerformanceTrainer
from ml.doctor_report import DoctorReportParser, create_sample_report, SAMPLE_REPORT
from game.game_engine import adapt_inputs, GameInputs
from core.fusion_engine import FusionEngine
from core.recovery_tracker import RecoveryTracker, init_db, save_session, load_session_history, print_recovery_report
from core.mqtt_publisher import MQTTPublisher


def parse_args():
    parser = argparse.ArgumentParser(description="NeuroRehab Demo")
    parser.add_argument("--game", default="bubble_pop",
                        choices=["bubble_pop", "flower_bloom", "pump_the_pump", "maze_steering"],
                        help="Which game to run")
    parser.add_argument("--pattern", default="moderate",
                        choices=["poor", "moderate", "good"],
                        help="Simulated EMG pattern")
    parser.add_argument("--duration", type=int, default=60,
                        help="Session duration in seconds")
    parser.add_argument("--patient-id", default=DEFAULT_PATIENT_ID)
    parser.add_argument("--pipeline-only", action="store_true",
                        help="Run pipeline test only (no game window)")
    parser.add_argument("--train", action="store_true",
                        help="Retrain model before running")
    return parser.parse_args()


def print_header():
    print("\n" + "═" * 60)
    print("  🧠  NeuroRehab — Smart Stroke Rehabilitation System")
    print("═" * 60)


def ensure_model(retrain: bool = False):
    """Ensure ML model exists, train if not."""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "data", "models", "performance_classifier.pkl")
    if retrain or not os.path.exists(model_path):
        print("\n[Setup] Training ML model on synthetic data...")
        trainer = PerformanceTrainer()
        meta = trainer.train(use_real_data=True, n_synthetic=2000)
        print(f"[Setup] Model trained — accuracy: {meta.get('accuracy', 0):.3f}")
    else:
        print("[Setup] ML model found ✓")


def ensure_sample_report():
    """Create sample doctor report if none exists."""
    report_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "data", "doctor_reports")
    if not any(f.endswith(".json") for f in os.listdir(report_dir) if os.path.exists(report_dir)):
        create_sample_report()
        print("[Setup] Sample doctor report created ✓")
    else:
        print("[Setup] Doctor report found ✓")


def pipeline_demo(args):
    """
    Run the full pipeline without a game window.
    Tests: EMG → Preprocessing → ML → Session logging
    """
    print("\n── Pipeline Integration Test ──────────────────────────")
    print(f"Patient: {args.patient_id}")
    print(f"EMG pattern: {args.pattern}")
    print(f"Duration: {args.duration}s")

    # Start EMG reader
    print("\n[EMG] Starting simulated reader...")
    reader = create_emg_reader("simulated", pattern=args.pattern)
    time.sleep(0.5)

    # Preprocessor
    preprocessor = EMGPreprocessor()

    # Vision tracker (simulated mode)
    tracker = MovementTracker(camera_index=0)
    tracker.start()

    # ML predictor
    predictor = PerformancePredictor()

    # Doctor report
    parser = DoctorReportParser()
    plan = parser.parse_dict(SAMPLE_REPORT)
    print(f"[Doctor] Therapy plan loaded for {plan.patient_id}")
    print(f"         Game sequence: {plan.get_game_sequence()}")

    # Session logger
    logger = SessionLogger(args.patient_id)

    # Progress tracker
    progress = ProgressTracker(args.patient_id)

    # Initialize fusion and recovery
    db_con = init_db()
    mqtt_pub = MQTTPublisher(MQTT_BROKER, MQTT_PORT, PATIENT_ID)
    
    # Wait for MQTT connection to establish
    time.sleep(1.0)
    print(f"[MQTT] Publisher connected: {mqtt_pub.connected}")
    
    fusion_eng = FusionEngine()
    rec_tracker = RecoveryTracker(PATIENT_ID, db_con, mqtt_pub)
    trainer = PerformanceTrainer()  # For online learning

    print(f"\n[Pipeline] Running for {args.duration}s ...\n")
    
    # Wait for EMG buffer to fill initially
    print("[Pipeline] Waiting for EMG buffer to fill...")
    time.sleep(1.0)

    start = time.time()
    window_count = 0
    labels = []
    last_mqtt = 0.0
    last_status_print = 0.0

    try:
        while time.time() - start < args.duration:
            # Collect EMG window - use blocking read to ensure we get data
            raw = np.array(reader.read(EMG_WINDOW_SIZE))
            if len(raw) < EMG_WINDOW_SIZE:
                print(f"[Pipeline] Warning: only got {len(raw)} samples, expected {EMG_WINDOW_SIZE}")
                time.sleep(0.05)
                continue

            raw_window = raw[-EMG_WINDOW_SIZE:]
            filtered, features = preprocessor.process_window(raw_window)

            # Vision (simulated)
            _, motion_frame = tracker.read_frame()

            # Fusion: combine EMG + vision
            finger_extensions = get_finger_extensions(motion_frame.hand)
            finger_moving = get_finger_moving(motion_frame.hand)
            
            # Convert EMGFeatures to dict for fusion engine
            emg_features_dict = {
                "rms": features.rms,
                "mav": features.mav,
                "zero_crossings": features.zc,
                "slope_sign_changes": features.ssc,
                "waveform_length": features.wl,
                "variance": features.var,
                "mean_freq": features.mean_freq,
                "median_freq": features.median_freq,
                "peak_amplitude": features.peak_amp,
                "contraction_ratio": features.contraction_ratio,
            }
            fusion_result = fusion_eng.classify(emg_features_dict, finger_extensions, finger_moving)

            # Log rep to recovery tracker (only if fusion produced a result)
            if fusion_result:
                rep = rec_tracker.log_rep(fusion_result)

                # Publish rep completion events (only when rep_completed)
                if fusion_result.get("rep_completed"):
                    mqtt_pub.publish("fusion", {
                        "state": fusion_result["state"],
                        "effort_pct": fusion_result["effort_pct"],
                        "rep_number": fusion_result["reps"],
                        "score_delta": fusion_result.get("score_delta", 0),
                        "rep_completed": True,
                        "ts": time.time(),
                    })
                    mqtt_pub.publish("game", fusion_eng.get_game_state())

                    # Get performance label from fusion state and connect to ML
                    perf_label = fusion_eng.get_performance_label()

                    if perf_label:
                        # Adjust therapy plan difficulty based on current performance
                        plan.adjust_difficulty(perf_label)

                        # Log labelled sample for future model retraining
                        trainer.log_labelled_sample(
                            emg_features = emg_features_dict,
                            motion       = {
                                "arm_angle" : motion_frame.pose.arm_angle,
                                "rom"       : motion_frame.rom,
                                "stability" : motion_frame.stability,
                                "velocity"  : motion_frame.pose.velocity,
                            },
                            performance_label = perf_label,
                            session_id        = logger.session_id,
                        )

                        # Retrain model in background if enough new data has accumulated
                        if trainer.should_retrain():
                            threading.Thread(
                                target=trainer.train,
                                kwargs={"use_real_data": True},
                                daemon=True
                            ).start()
                            print("[ML] Background retraining started — new real data available")

            # Continuous MQTT streaming (every 50ms, regardless of fusion state)
            now = time.time()
            if now - last_mqtt > 0.05:  # 50ms rate limit
                mqtt_pub.publish("emg", {
                    "rms": features.rms,
                    "effort_pct": round(features.contraction_ratio * 100, 1),
                    "contracting": features.contraction_ratio >= 0.35,
                    "median_freq": features.median_freq,
                    "fatigue_alert": fusion_result.get("fatigue_alert", False) if fusion_result else False,
                    "ts": now,
                })
                # Continuous fusion state
                mqtt_pub.publish("fusion", {
                    "state": fusion_result["state"] if fusion_result else "REST",
                    "effort_pct": fusion_result.get("effort_pct", 0) if fusion_result else 0,
                    "rep_completed": False,
                    "ts": now,
                })
                # Handpose data
                if motion_frame and motion_frame.hand:
                    extensions = get_finger_extensions(motion_frame.hand)
                    moving = get_finger_moving(motion_frame.hand)
                    mqtt_pub.publish("handpose", {
                        "extensions": extensions,
                        "moving": moving,
                        "confidence": getattr(motion_frame.hand, "confidence", 0.9),
                        "ts": now,
                    })
                last_mqtt = now

            # ML prediction
            prediction = predictor.predict(
                features,
                arm_angle=motion_frame.pose.arm_angle,
                rom=motion_frame.rom,
                stability=motion_frame.stability,
                velocity=motion_frame.pose.velocity,
            )

            # Adapt difficulty based on doctor plan (using ML prediction)
            plan.adjust_difficulty(prediction["label"])

            # Log
            logger.log(
                features,
                arm_angle=motion_frame.pose.arm_angle,
                rom=motion_frame.rom,
                stability=motion_frame.stability,
                velocity=motion_frame.pose.velocity,
                game_name=plan.get_game_sequence()[0],
                score=float(prediction["score"]),
                difficulty=plan.exercises[0].difficulty if plan.exercises else "easy",
                performance_label=prediction["label"],
            )

            labels.append(prediction["label"])
            window_count += 1

            # Print status every 10 windows
            if window_count % 10 == 0:
                elapsed = time.time() - start
                label_counts = {l: labels.count(l) for l in set(labels)}
                print(
                    f"  t={elapsed:5.1f}s  |  label={prediction['label']:12s}  "
                    f"|  score={prediction['score']:3d}  |  rms={features.rms:.4f}  "
                    f"|  ROM={motion_frame.rom:.1f}°"
                )

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by user")

    finally:
        reader.stop()
        tracker.stop()
        summary = logger.close()
        progress.record_session(summary)

        # Save recovery session
        session_metrics = rec_tracker.build_metrics()
        session_id, saved_num, mvc_norm, baseline_mvc = save_session(
            db_con, PATIENT_ID, session_metrics, rec_tracker.rep_log
        )

        # Publish session summary
        mqtt_pub.publish("session_summary", {
            **session_metrics,
            "session_number": saved_num,
            "mvc_normalised": mvc_norm,
            "baseline_mvc": baseline_mvc,
            "patient_id": PATIENT_ID,
            "ts": time.time(),
        })

        # Print recovery report
        history = load_session_history(db_con, PATIENT_ID)
        print_recovery_report(history, PATIENT_ID)
        db_con.close()

        print("\n── Session Summary ────────────────────────────────────")
        print(f"  Total windows:  {window_count}")
        print(f"  Mean RMS:       {summary.get('emg', {}).get('mean_rms', 0):.4f}")
        print(f"  Mean ROM:       {summary.get('motion', {}).get('mean_rom', 0):.1f}°")
        print(f"  Mean score:     {summary.get('game', {}).get('mean_score', 0):.1f}")
        print(f"  Performance:    {summary.get('game', {}).get('dominant_performance', 'N/A')}")
        print(f"\n  Session file:   {summary.get('csv_path', 'N/A')}")

        trend = progress.get_trend()
        if trend.get("total_sessions", 0) > 1:
            print(f"\n── Recovery Trend ({trend['total_sessions']} sessions) ───────────────")
            direction = "↑ Improving" if trend["score_trend"] > 0 else "↓ Declining"
            print(f"  Score trend:    {direction} ({trend['score_trend']:+.2f}/session)")
            print(f"  ROM trend:      {trend['rom_trend']:+.2f}°/session")
            print(f"  Best score:     {trend['best_score']:.1f}")


def game_demo(args):
    """Launch a game with live EMG + motion pipeline."""
    print(f"\n[Game] Launching: {args.game}")
    print("[Game] Controls:")
    print("  • Mouse position → wrist angle / arm elevation")
    print("  • Left click / SPACE → grip/pinch action")
    print("  • ESC → quit")
    print("\n  (In a real session, EMG sensor + webcam control the game)")
    print("\nStarting in 2 seconds...")
    time.sleep(2)

    # Start EMG
    reader = create_emg_reader("simulated", pattern=args.pattern)
    preprocessor = EMGPreprocessor()
    tracker = MovementTracker()
    tracker.start()
    predictor = PerformancePredictor()
    logger = SessionLogger(args.patient_id)
    time.sleep(0.4)

    # Shared state for input thread
    shared = {"inputs": GameInputs(), "running": True}

    def input_thread():
        """Background thread: reads EMG+vision, updates shared inputs."""
        while shared["running"]:
            raw = np.array(reader.read_available())
            if len(raw) >= EMG_WINDOW_SIZE:
                _, features = preprocessor.process_window(raw[-EMG_WINDOW_SIZE:])
                _, motion = tracker.read_frame()
                prediction = predictor.predict(
                    features,
                    arm_angle=motion.pose.arm_angle,
                    rom=motion.rom,
                    stability=motion.stability,
                    velocity=motion.pose.velocity,
                )
                inp = adapt_inputs(features, motion, prediction)
                shared["inputs"] = inp
                logger.log(
                    features,
                    arm_angle=motion.pose.arm_angle,
                    rom=motion.rom,
                    stability=motion.stability,
                    score=float(prediction["score"]),
                    game_name=args.game,
                    performance_label=prediction["label"],
                )
            time.sleep(0.05)

    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    # Launch game
    try:
        if args.game == "bubble_pop":
            from game.bubble_pop import BubblePopGame
            game = BubblePopGame(session_duration=args.duration)
            score = game.run_loop(input_callback=lambda: shared["inputs"])
        elif args.game == "flower_bloom":
            from game.flower_bloom import FlowerBloomGame
            game = FlowerBloomGame(session_duration=args.duration)
            score = game.run_loop(input_callback=lambda: shared["inputs"])
        elif args.game == "pump_the_pump":
            from game.pump_maze import PumpThePumpGame
            game = PumpThePumpGame(session_duration=args.duration)
            score = game.run_loop(input_callback=lambda: shared["inputs"])
        elif args.game == "maze_steering":
            from game.pump_maze import MazeSteeringGame
            game = MazeSteeringGame(session_duration=args.duration)
            score = game.run_loop(input_callback=lambda: shared["inputs"])
        else:
            print(f"Unknown game: {args.game}")
            score = 0

        print(f"\n[Game] Final score: {score}")
    finally:
        shared["running"] = False
        reader.stop()
        tracker.stop()
        summary = logger.close()
        print(f"[Session] Saved: {summary.get('csv_path', 'N/A')}")


def main():
    args = parse_args()
    print_header()

    # Setup
    print("\n── System Setup ────────────────────────────────────────")
    ensure_model(retrain=args.train)
    ensure_sample_report()

    if args.pipeline_only:
        pipeline_demo(args)
    else:
        try:
            import pygame
            game_demo(args)
        except ImportError:
            print("\n[Warning] pygame not installed — running pipeline test only")
            print("  Install with: pip install pygame")
            pipeline_demo(args)

    print("\n✅ Demo complete.\n")


if __name__ == "__main__":
    main()
