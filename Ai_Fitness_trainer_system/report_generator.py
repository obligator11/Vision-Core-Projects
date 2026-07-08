

import csv
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import CALORIES_PER_REP, CALORIES_PER_SECOND_HOLD


def estimate_calories(counters, plank_seconds):
    total = 0.0
    for ex, reps in counters.items():
        total += reps * CALORIES_PER_REP.get(ex, 0.25)
    total += plank_seconds * CALORIES_PER_SECOND_HOLD.get("Plank", 0.045)
    return round(total, 1)


def save_session_report(counters, plank_seconds, form_score, duration_sec, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    calories = estimate_calories(counters, plank_seconds)

    # --- CSV log (append-friendly, one row per session) ---
    csv_path = os.path.join(out_dir, "workout_history.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["timestamp", "duration_sec", "squats", "pushups", "lunges",
                 "jumping_jacks", "plank_seconds", "form_score_pct", "calories_kcal"]
            )
        writer.writerow([
            timestamp, round(duration_sec, 1),
            counters.get("Squat", 0), counters.get("Push-up", 0),
            counters.get("Lunge", 0), counters.get("Jumping Jack", 0),
            round(plank_seconds, 1), form_score, calories,
        ])

    # --- Bar chart summary for this session ---
    labels = ["Squats", "Push-ups", "Lunges", "Jump. Jacks", "Plank (s)"]
    values = [
        counters.get("Squat", 0), counters.get("Push-up", 0),
        counters.get("Lunge", 0), counters.get("Jumping Jack", 0),
        round(plank_seconds, 1),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=["#4C9AFF", "#36B37E", "#FFAB00", "#6554C0", "#FF5630"])
    ax.set_title(f"Workout Summary — {timestamp}\nForm Score: {form_score}%  |  ~{calories} kcal burned")
    ax.set_ylabel("Reps / Seconds")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(val), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    chart_path = os.path.join(out_dir, f"session_{timestamp}.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    return {
        "csv_path": csv_path,
        "chart_path": chart_path,
        "calories": calories,
        "timestamp": timestamp,
    }