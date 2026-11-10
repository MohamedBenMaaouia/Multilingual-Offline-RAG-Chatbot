"""Day 2 gate: end-to-end ingest -> retrieve -> answer with sources.

Generates dummy data into data/raw/ if it's empty (a 2-page automotive PDF
and an OBD codes CSV), then runs the full pipeline and asserts:
  - the answer is non-empty
  - at least one retrieved source ends with 'obd_codes.csv'

Run from project root:

    python scripts/smoke_test.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# Make sure `src.*` is importable when run as a plain script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402


# --------------------------------------------------------------------------
# Dummy data generation
# --------------------------------------------------------------------------

OIL_PARAGRAPH = (
    "Regular engine oil maintenance is one of the most important services for "
    "a modern passenger vehicle. For most gasoline-engine cars manufactured "
    "after 2010, the manufacturer recommends changing the engine oil and the "
    "oil filter every 10,000 km or every 6 months, whichever comes first. "
    "Vehicles operating under severe conditions, such as frequent short trips, "
    "stop-and-go traffic, dusty environments, or towing, should follow the "
    "severe-service schedule and change the oil every 5,000 km or 3 months. "
    "Use only the viscosity and specification grade listed in the owner's "
    "manual, typically 5W-30 or 0W-20 synthetic for modern engines. After "
    "draining the old oil and replacing the filter, refill with the exact "
    "quantity specified by the manufacturer, usually between 3.5 and 4.5 "
    "liters for a four-cylinder engine. Run the engine for one minute, shut "
    "it off, wait five minutes, and verify the level with the dipstick. "
    "Top up if the level sits below the MIN mark. Never overfill, as excess "
    "oil can be aerated by the crankshaft and damage seals. Dispose of used "
    "oil at an authorized collection point; never pour it down a drain. "
    "Always reset the oil-life monitor in the instrument cluster after the "
    "service so that the next interval is computed from the correct baseline."
)

BRAKE_PARAGRAPH = (
    "Brake pad inspection should be performed at every routine service "
    "interval and any time the driver notices unusual noises, vibration, or "
    "a longer pedal travel. Most modern passenger cars use disc brakes on "
    "the front axle and either disc or drum brakes on the rear axle. To "
    "inspect the front pads, raise the vehicle on a lift or jack stands, "
    "remove the wheel, and visually estimate the remaining friction material "
    "thickness through the caliper window. New pads start at approximately "
    "10 to 12 millimeters of friction material; the manufacturer wear limit "
    "is typically 3 millimeters. Below 3 millimeters the pads should be "
    "replaced as a complete axle set, even if only one side is worn. Inspect "
    "the brake disc surface at the same time for grooves, blue heat marks, "
    "or excessive lip on the outer edge. Replace discs when the thickness "
    "falls below the minimum stamped on the hub face. After replacing pads, "
    "bed them in with twenty moderate stops from 60 km/h to 10 km/h without "
    "coming to a full stop, then allow them to cool. Skipping the bed-in "
    "procedure can lead to uneven pad transfer and brake judder. Verify "
    "fluid level in the reservoir after the job, since pushing the caliper "
    "pistons back will raise the level."
)


def _write_dummy_pdf(out_path: Path) -> None:
    """Generate a two-page automotive PDF via reportlab."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    title = styles["Title"]

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        title="Dummy Service Manual",
    )
    flow = [
        Paragraph("Section 1 - Engine Oil Service", title),
        Spacer(1, 12),
        Paragraph(OIL_PARAGRAPH, body),
        PageBreak(),
        Paragraph("Section 2 - Brake System Inspection", title),
        Spacer(1, 12),
        Paragraph(BRAKE_PARAGRAPH, body),
    ]
    doc.build(flow)


OBD_ROWS = [
    ("P0420", "Catalyst System Efficiency Below Threshold (Bank 1)", "Emissions", "Moderate"),
    ("P0301", "Cylinder 1 Misfire Detected", "Engine", "High"),
    ("P0171", "System Too Lean (Bank 1)", "Fuel", "Moderate"),
    ("P0455", "Evaporative Emission System Leak Detected (Large)", "Emissions", "Low"),
    ("P0128", "Coolant Thermostat Below Regulating Temperature", "Cooling", "Low"),
    ("P0300", "Random/Multiple Cylinder Misfire Detected", "Engine", "High"),
    ("P0442", "Evaporative Emission System Leak Detected (Small)", "Emissions", "Low"),
    ("P0507", "Idle Air Control System RPM Higher Than Expected", "Engine", "Moderate"),
    ("P0606", "Powertrain Control Module Processor Fault", "PCM", "High"),
    ("P0700", "Transmission Control System Malfunction", "Transmission", "High"),
]


def _write_dummy_csv(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["code", "description", "system", "severity"])
        writer.writerows(OBD_ROWS)


def _real_data_present(raw: Path) -> bool:
    """Detect whether the user has already placed real data under data/raw/.

    Heuristic: any PDF other than dummy_manual.pdf, OR an obd_codes.csv with
    more than 50 lines, counts as "real data" — skip dummy generation in
    that case so we don't accidentally mix dummies into a real corpus.
    """
    manuals = raw / "manuals"
    if manuals.exists():
        for pdf in manuals.rglob("*.pdf"):
            if pdf.name != "dummy_manual.pdf":
                return True
    csv_path = raw / "obd_codes.csv"
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
                # Count up to 51 lines — cheap upper bound.
                n = sum(1 for _ in zip(range(51), fh))
            if n > 50:
                return True
        except OSError:
            pass
    return False


def ensure_dummy_data() -> None:
    """Materialize a dummy corpus only when no real data is present."""
    raw = settings.data_raw_path
    raw.mkdir(parents=True, exist_ok=True)

    if _real_data_present(raw):
        print(f"      Real data detected in {raw}, skipping dummy generation.")
        return

    pdf_path = raw / "manuals" / "dummy_manual.pdf"
    csv_path = raw / "obd_codes.csv"

    if not pdf_path.exists():
        print(f"      generating {pdf_path}")
        _write_dummy_pdf(pdf_path)
    else:
        print(f"      {pdf_path} already exists, skipping")

    if not csv_path.exists():
        print(f"      generating {csv_path}")
        _write_dummy_csv(csv_path)
    else:
        print(f"      {csv_path} already exists, skipping")


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------

QUERY = "What does code P0420 mean?"


def truncate(text: str, n: int = 200) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def main() -> int:
    print("[1/5] Ensuring dummy corpus exists under data/raw/...")
    ensure_dummy_data()

    print("[2/5] Running ingest(wipe_first=True)...")
    from src.ingest import ingest

    ingest_result = ingest(wipe_first=True)
    print(f"      ingest result: {ingest_result}")

    # Sanity check the collection has rows.
    from src.vectorstore import count_chunks

    coll_count = count_chunks()
    print(f"      collection count: {coll_count}")
    if coll_count == 0:
        print("[FAIL] collection is empty after ingest")
        return 1

    print("[3/5] Building the RAG chain (cold-loads Ollama)...")
    from src.chain import get_chain

    chain = get_chain()

    print(f"[4/5] Invoking chain with query: {QUERY!r}")
    t0 = time.perf_counter()
    result = chain.invoke({"input": QUERY})
    latency_ms = int((time.perf_counter() - t0) * 1000)

    answer = result.get("answer", "")
    context_docs = result.get("context", []) or []

    print("[5/5] Results:")
    print(f"      latency: {latency_ms} ms")
    print(f"      retrieved_k: {len(context_docs)}")
    print("      --- answer ---")
    print(answer)
    print("      --- sources ---")
    for i, d in enumerate(context_docs, start=1):
        md = d.metadata or {}
        page = md.get("page")
        page_display = None if page in (-1, None) else page
        print(
            f"      [{i}] source={md.get('source')!r}  "
            f"page={page_display}  doc_type={md.get('doc_type')!r}"
        )
        print(f"           snippet: {truncate(d.page_content)!r}")

    # ---- Gate assertions ----
    answer_ok = bool(answer and answer.strip())
    obd_in_sources = any(
        (d.metadata or {}).get("source", "").endswith("obd_codes.csv")
        for d in context_docs
    )

    print()
    print(f"      assertion: answer non-empty -> {answer_ok}")
    print(f"      assertion: obd_codes.csv in sources -> {obd_in_sources}")

    if not (answer_ok and obd_in_sources):
        print("[FAIL] Day 2 gate not met (see assertions above)")
        return 1

    print("[OK] Day 2 gate: passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
