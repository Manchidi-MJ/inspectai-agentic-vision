# Responsible Use

InspectAI is **decision-support tooling**, not a certified quality-control
system. Read this before using it on anything that matters.

## What InspectAI does and does not claim

- **Does:** localize *visual anomalies* — compact regions that deviate in
  contrast from their local surroundings — using classical OpenCV analysis, and
  route uncertain cases to human review.
- **Does not:** confirm defects, guarantee detection of all real defects, or
  produce calibrated probabilities. A flagged region is a **potential** visual
  anomaly, nothing more.

Three distinct outcomes are kept separate on purpose:
*potential anomaly* ≠ *confirmed defect* ≠ *uncertain / human review*.

## Human-in-the-loop requirement

Every automated finding requires verification by a qualified human inspector
before any accept/reject/scrap decision. The UI and API surface a human-review
status and display this warning. Do not wire InspectAI directly into an
automated reject actuator without a human gate.

## Thresholds are not probabilities

All thresholds (quality flags, score bands) are documented **heuristics** tuned
to a controlled-lighting assumption. They are not calibrated confidences and
must be re-validated for each product, surface, and camera setup.

## Data handling and privacy

- The app processes images provided by the operator. It does **not** persist
  uploads server-side beyond the request, and does **not** transmit images to
  any third party.
- No credentials or secrets are stored in source; cloud deployment uses IAM
  roles and non-secret environment variables only.
- If used on images that may contain people or sensitive content, the operator
  is responsible for complying with applicable privacy law and obtaining any
  necessary consent. InspectAI is designed for *manufactured-object surfaces*,
  not for identifying people.

## Safety posture

- A poor-quality image is **never** declared defective — it is escalated.
- Inconclusive analysis results in explicit uncertainty (human review), never a
  silent pass.
- The agent loop is bounded; it cannot spin indefinitely.

## Known limitations

See `docs/evaluation.md` §"Failure case" and the technical report §10. In
short: synthetic evaluation data, a classical method with no semantic
understanding, and a documented false-negative on faint low-contrast defects.
