# InspectAI — 5-minute Demo Script & Recording Checklist

> **No video exists yet.** This is the script to record one. Nothing here should
> be presented as already filmed until it has been.

Target length: ~3–5 minutes. Record the Streamlit UI (`streamlit run app.py`)
at 1080p. Have the evaluation chart (`artifacts/score_distribution.png`) and
the architecture diagram (`deployment/architecture.md`) open in tabs.

## Beat-by-beat (with suggested narration)

**0:00–0:30 — Problem & pitch**
> "Automated inspection usually spits out a score and moves on. It never
> decides when the evidence is too weak to trust, or when a human should look.
> InspectAI localizes visual anomalies with OpenCV 5, then runs an agent that
> investigates its own evidence and decides what happens next."

**0:30–1:15 — Normal sample**
- Pick sample kind **normal**, seed 101 → **Inspect image**.
- Show status **✅ No anomaly detected**, low score, empty/clean heatmap.
> "Clean surface. Low anomaly score, no coherent region — the agent terminates
> with a pass in two steps. Note there's no false alarm."

**1:15–2:15 — Clear defect (spot)**
- Pick sample kind **spot**, seed 303 → **Inspect image**.
- Show status **⚠️ Potential anomaly**, high score (~0.97), the annotated box
  and bright heatmap blob, and the region measurements table.
> "Here's a contamination spot. OpenCV's high-pass residual lights it up, the
> score is high with one coherent region, so the agent terminates immediately
> as a potential anomaly — and shows the measured bounding box, area, and
> contrast."

**2:15–3:15 — The agentic moment (faint scratch)**
- Pick sample kind **scratch**, **seed 102** → Inspect (score ~0.44, borderline,
  re-inspects, does **not** corroborate → **human review**).
- Then **seed 111** → Inspect (score ~0.51, borderline, re-inspection **does**
  corroborate → **potential anomaly**). Two different actions, two different
  conclusions, both driven by the first result.
- Open the **Agent decision trace** expanders to show the choose → reinspect →
  compare steps.
> "This faint scratch is borderline. Watch the trace: the agent's FIRST result
> is in the grey zone, so it CHOOSES a different action — a targeted
> re-inspection with a lowered threshold — then compares the two. The evidence
> doesn't corroborate, so instead of guessing it routes to **human review**.
> The next action depended on the first visual result. That's the agentic loop."

**3:15–4:00 — Evidence & honesty**
- Show `artifacts/score_distribution.png` and the metrics table.
> "On a reproducible synthetic set: ROC-AUC 0.994, F1 0.987, zero false
> positives, one false negative — a faint scratch we miss. We report that
> failure case rather than hide it. The data is synthetic and the thresholds
> are heuristics, not calibrated probabilities."

**4:00–4:40 — AWS component**
- Show `deployment/architecture.md` diagram and the FastAPI `/health` +
  `/inspect` (if deployed, hit the live App Runner URL; otherwise show the
  local `uvicorn` run and state clearly it is not yet deployed).
> "The same pipeline runs as a FastAPI service, containerized for AWS App
> Runner — image upload in, structured results and annotated images out, with a
> health endpoint. [If live: here it is on AWS. If not: this runs locally today;
> deployment steps are ready but unverified.]"

**4:40–5:00 — Close**
> "InspectAI: OpenCV 5 for the seeing, an explicit agent for the reasoning and
> deciding, and a human kept in the loop wherever the evidence is uncertain."

## Recording checklist

- [ ] `pip install -r requirements.txt` done in a clean env.
- [ ] `python scripts/make_samples.py` run (stable sample files).
- [ ] `python -m src.evaluation.evaluate` run (fresh `evaluation_results.json`).
- [ ] `python scripts/make_charts.py` run (fresh charts).
- [ ] Grey-zone scratch seeds confirmed: **102** (→ human review), **111**
      (→ potential anomaly after corroboration). (Regenerate if the detector
      is retuned.)
- [ ] `streamlit run app.py` up and responsive.
- [ ] (Optional) App Runner service deployed and `/health` returns 200; capture
      the URL. If not deployed, say so on camera.
- [ ] Screen recording at 1080p, audio levels checked.
- [ ] Final video ≤ 5 minutes.
