"""Train an experimental, single-participant WiSense posture classifier.

Run: py wisense_ml_train.py --csv wisense_posture_raw.csv
The newest complete trial of EACH label is reserved for testing.
Never tune the model repeatedly against this test set; collect fresh trials.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = ["STANDING", "MOVING"]
COLS = [f"sc_{i:02d}" for i in range(64)]
WINDOW = 12


def features(amplitudes):
    """Input: 12 x selected-bin normalized amplitudes; reuse for live inference."""
    x = np.asarray(amplitudes, dtype=float)
    return np.concatenate([x.mean(axis=0), x.std(axis=0),
                           np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0),
                           np.abs(np.diff(x, axis=0)).mean(axis=0)])


def windows(trials, ids, bins):
    xs, ys, records = [], [], []
    for sid in ids:
        g = trials[sid]
        a = g[COLS].to_numpy(float)[:, bins]
        # Non-overlapping windows; no window spans a recording boundary.
        for start in range(0, len(g) - WINDOW + 1, WINDOW):
            t = g.iloc[start:start + WINDOW]
            gaps = t.timestamp.diff().dt.total_seconds().dropna()
            if (gaps <= 0).any() or (gaps > 1).any():
                continue
            xs.append(features(a[start:start + WINDOW]))
            ys.append(g.label.iloc[0])
            records.append({"session_id": sid, "start_frame": start,
                            "actual": g.label.iloc[0]})
    if not xs:
        raise ValueError("No valid windows. Check timestamps and CSI input.")
    return np.array(xs), np.array(ys), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(Path(__file__).with_name("wisense_posture_raw.csv")))
    parser.add_argument("--evaluate-only", action="store_true", help="Print results without saving files")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    df = df[df["label"].isin(LABELS)].copy()
    required = ["timestamp", "session_id", "participant", "label"] + COLS
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[required].isna().any().any():
        raise ValueError("Missing values found; inspect the CSV before training.")
    df[COLS] = df[COLS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[COLS].to_numpy()).all():
        raise ValueError("Non-finite CSI values found.")
    df.timestamp = pd.to_datetime(df.timestamp, errors="raise")
    if set(df.label) != set(LABELS):
        raise ValueError(f"Expected exactly these labels: {LABELS}")
    if df.participant.nunique() != 1:
        raise ValueError("This starter split is for one participant. Multiple people need a participant-based split.")

    trials, excluded = {}, []
    for sid, g in df.groupby("session_id"):
        if g.label.nunique() != 1:
            raise ValueError(f"Mixed labels in session {sid}")
        g = g.sort_values("timestamp").reset_index(drop=True)
        seconds = (g.timestamp.iloc[-1] - g.timestamp.iloc[0]).total_seconds()
        if seconds < 18 or len(g) < 80:
            excluded.append(str(sid))
            continue
        trials[sid] = g

    train_ids, test_ids = [], []
    for label in LABELS:
        ids = sorted([s for s, g in trials.items() if g.label.iloc[0] == label],
                     key=lambda s: trials[s].timestamp.iloc[0])
        if len(ids) < 5:
            raise ValueError(f"{label}: need 5 complete trials; found {len(ids)}.")
        train_ids.extend(ids[:-1])
        test_ids.append(ids[-1])
    assert set(train_ids).isdisjoint(test_ids)
    # Select bins only from training recordings, never the test data.
    raw_train = pd.concat([trials[s] for s in train_ids])[COLS].to_numpy(float)
    bins = np.flatnonzero((raw_train > 0).all(axis=0) & (raw_train.std(axis=0) > 1e-8))
    if len(bins) < 10:
        raise ValueError("Fewer than 10 consistently usable CSI bins.")
    x_train, y_train, _ = windows(trials, train_ids, bins)
    x_test, y_test, records = windows(trials, test_ids, bins)
    model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                   min_samples_leaf=3, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    report = classification_report(y_test, prediction, labels=LABELS, zero_division=0)
    matrix = pd.DataFrame(confusion_matrix(y_test, prediction, labels=LABELS),
                          index=LABELS, columns=LABELS)
    print(f"Trials: {len(train_ids)} train / {len(test_ids)} test; excluded: {len(excluded)}")
    print(f"CSI bins: {len(bins)}; non-overlapping windows: {len(y_train)} train / {len(y_test)} test")
    print(f"\nHeld-out window accuracy: {accuracy:.1%}\n")
    print(report)
    print("Confusion matrix: ROW = actual, COLUMN = predicted")
    print(matrix.to_string())
    print("\nExperimental SAME-PERSON, SAME-SETUP result. Windows within trials are correlated.")
    print("Fresh live trials are still needed; this is not proof of general posture recognition.")
    if args.evaluate_only:
        return
    out = Path(__file__).resolve().parent / ("ml_run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir()
    bundle = {"model": model, "bin_indices": bins, "window_size": WINDOW,
              "feature_version": "mean_std_iqr_absdiff_v1", "columns": COLS,
              "normalization": "collector_per_packet_positive_median_first64",
              "sklearn_version": sklearn.__version__}
    joblib.dump(bundle, out / "wisense_posture_model.joblib")
    for row, pred in zip(records, prediction):
        row["predicted"] = pred
    pd.DataFrame(records).to_csv(out / "test_predictions.csv", index=False)
    details = {"train_sessions": train_ids, "test_sessions": test_ids,
               "excluded_sessions": excluded, "accuracy": accuracy,
               "bin_indices": bins.tolist(), "window_frames": WINDOW,
               "sklearn_version": sklearn.__version__}
    (out / "split_and_metrics.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (out / "report.txt").write_text(report + "\n" + matrix.to_string(), encoding="utf-8")
    print(f"\nSaved model and reports in: {out}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"Training stopped: {error}")
