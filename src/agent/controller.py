"""Agentic inspection controller.

An explicit, bounded perceive -> evaluate -> choose -> act -> observe loop. The
decision at each step is a pure function of the real measurements produced by
the OpenCV engine, so the trace demonstrably shows the first visual result
steering the next action.

Decision policy (documented, deterministic):
  * If image quality fails -> escalate to human review (a bad image is NEVER
    declared defective).
  * Else run analysis.
      - Clear anomaly (score >= HIGH and >=1 region) -> potential anomaly.
      - Clearly clean (score <= LOW and 0 regions) -> no anomaly detected.
      - Borderline -> the agent CHOOSES to re-inspect with a lowered threshold
        (a different action driven by the first result), then:
          * if re-inspection corroborates (regions appear / score rises enough)
            -> potential anomaly;
          * else -> human review (uncertain), never a silent pass.
  * Iterations are bounded; hitting the cap yields human review, not a guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.agent import tools
from src.agent.decision_log import DecisionLog, Step
from src.vision import detector
from src.vision.preprocessing import Prepared


# Documented thresholds for the policy (heuristics tuned on the synthetic
# dataset, see docs/evaluation.md; NOT calibrated probabilities).
# On the dataset: normal scores <= 0.48, defect scores in [0.43, 0.97].
SCORE_HIGH = 0.55      # at/above => treat as a clear potential anomaly
SCORE_LOW = 0.50       # below (with no regions) => treat as clean
REINSPECT_K = 2.0      # lowered threshold-k used on borderline re-inspection
CORROBORATE_DELTA = 0.05  # min score rise on re-inspection to corroborate


class Status(str, Enum):
    POTENTIAL_ANOMALY = "potential_anomaly"
    NO_ANOMALY = "no_anomaly_detected"
    HUMAN_REVIEW = "human_review_required"


@dataclass
class InspectionOutcome:
    status: Status
    anomaly_score: Optional[float]
    num_regions: int
    quality: Dict
    detection: Optional[Dict]
    regions: List[Dict] = field(default_factory=list)
    decision_log: List[Dict] = field(default_factory=list)
    iterations: int = 0
    escalated: bool = False

    def to_dict(self) -> Dict:
        return {
            "status": self.status.value,
            "anomaly_score": self.anomaly_score,
            "num_regions": self.num_regions,
            "quality": self.quality,
            "detection": self.detection,
            "regions": self.regions,
            "decision_log": self.decision_log,
            "iterations": self.iterations,
            "escalated": self.escalated,
        }


class InspectionAgent:
    def __init__(self, max_iters: int = 4) -> None:
        self.max_iters = max_iters

    def run(self, prep: Prepared) -> InspectionOutcome:
        log = DecisionLog()
        it = 0

        # ---- Phase: perceive + evaluate quality ----
        it += 1
        quality = tools.assess_image_quality(prep)
        log.add(Step(
            iteration=it, phase="evaluate",
            observation={"quality": quality},
            action="assess_image_quality", action_params={},
            rationale="Gate on image quality before trusting any detection.",
            result={"usable": quality["usable"], "flags": quality["flags"]},
        ))

        if not quality["usable"]:
            res = tools.request_human_review(
                reason=f"Image quality insufficient: {', '.join(quality['flags'])}. "
                       f"A degraded image is not declared defective."
            )
            log.add(Step(
                iteration=it, phase="act",
                observation={"flags": quality["flags"]},
                action="request_human_review",
                action_params={"reason": res["reason"]},
                rationale="Poor-quality input cannot support a reliable decision.",
                result=res,
            ))
            return InspectionOutcome(
                status=Status.HUMAN_REVIEW, anomaly_score=None, num_regions=0,
                quality=quality, detection=None, regions=[],
                decision_log=log.to_list(), iterations=it, escalated=True,
            )

        # ---- Phase: perceive (analyze) ----
        it += 1
        first = tools.analyze_image(prep)
        log.add(Step(
            iteration=it, phase="perceive",
            observation={"stage": "primary_analysis"},
            action="analyze_image",
            action_params={"thresh_k": detector.THRESH_K},
            rationale="Run default anomaly localization on the usable image.",
            result=first.to_dict(),
        ))

        score = first.anomaly_score
        n_regions = len(first.regions)

        # ---- Phase: choose (the branch depends on the real first result) ----
        if score >= SCORE_HIGH and n_regions >= 1:
            log.add(Step(
                iteration=it, phase="choose",
                observation={"anomaly_score": score, "num_regions": n_regions},
                action="terminate",
                rationale=f"Score {score:.2f} >= {SCORE_HIGH} with {n_regions} region(s): "
                          f"clear potential anomaly; no further analysis needed.",
                result={"status": Status.POTENTIAL_ANOMALY.value},
            ))
            return self._finish(Status.POTENTIAL_ANOMALY, first, quality, log, it)

        if score < SCORE_LOW:
            # Low image-level contrast: any regions present are low-coherence
            # speckle that the score already discounted. Clean pass.
            log.add(Step(
                iteration=it, phase="choose",
                observation={"anomaly_score": score, "num_regions": n_regions},
                action="terminate",
                rationale=f"Score {score:.2f} < {SCORE_LOW}: low, incoherent contrast; "
                          f"no anomaly detected by this system.",
                result={"status": Status.NO_ANOMALY.value},
            ))
            return self._finish(Status.NO_ANOMALY, first, quality, log, it)

        # ---- Borderline: the agent CHOOSES a different action ----
        if it >= self.max_iters:
            return self._escalate_cap(first, quality, log, it)

        log.add(Step(
            iteration=it, phase="choose",
            observation={"anomaly_score": score, "num_regions": n_regions},
            action="run_targeted_reinspection",
            action_params={"thresh_k": REINSPECT_K},
            rationale=f"Borderline result (score {score:.2f} in grey zone). "
                      f"Re-inspect with a lower threshold to probe weak evidence.",
            result={"planned": True},
        ))

        it += 1
        second = tools.run_targeted_reinspection(prep, thresh_k=REINSPECT_K)
        cmp = tools.compare_analysis_results(first, second)
        log.add(Step(
            iteration=it, phase="act",
            observation={"first_score": score, "second_score": second.anomaly_score},
            action="run_targeted_reinspection",
            action_params={"thresh_k": REINSPECT_K},
            rationale="Lowered-threshold re-inspection to corroborate or dismiss.",
            result={"detection": second.to_dict(), "comparison": cmp},
        ))

        corroborated = (cmp["score_delta"] >= CORROBORATE_DELTA) or \
                       (len(second.regions) > len(first.regions) and second.anomaly_score >= SCORE_LOW)

        if corroborated:
            log.add(Step(
                iteration=it, phase="terminate",
                observation={"comparison": cmp},
                action="terminate",
                rationale="Re-inspection corroborated weak evidence: potential anomaly.",
                result={"status": Status.POTENTIAL_ANOMALY.value},
            ))
            return self._finish(Status.POTENTIAL_ANOMALY, second, quality, log, it)

        res = tools.request_human_review(
            reason="Borderline anomaly evidence that re-inspection did not corroborate; "
                   "routing to human review rather than guessing."
        )
        log.add(Step(
            iteration=it, phase="terminate",
            observation={"comparison": cmp},
            action="request_human_review", action_params={"reason": res["reason"]},
            rationale="Inconclusive evidence must surface as uncertainty, not a pass.",
            result=res,
        ))
        return InspectionOutcome(
            status=Status.HUMAN_REVIEW, anomaly_score=second.anomaly_score,
            num_regions=len(second.regions), quality=quality,
            detection=second.to_dict(), regions=[r.to_dict() for r in second.regions],
            decision_log=log.to_list(), iterations=it, escalated=True,
        )

    def _escalate_cap(self, det, quality, log, it) -> InspectionOutcome:
        res = tools.request_human_review(reason="Iteration limit reached without a confident decision.")
        log.add(Step(
            iteration=it, phase="terminate",
            observation={"iterations": it, "max_iters": self.max_iters},
            action="request_human_review", action_params={"reason": res["reason"]},
            rationale="Bounded loop hit its cap; escalate instead of guessing.",
            result=res,
        ))
        return InspectionOutcome(
            status=Status.HUMAN_REVIEW, anomaly_score=det.anomaly_score,
            num_regions=len(det.regions), quality=quality, detection=det.to_dict(),
            regions=[r.to_dict() for r in det.regions],
            decision_log=log.to_list(), iterations=it, escalated=True,
        )

    @staticmethod
    def _finish(status, det, quality, log, it) -> InspectionOutcome:
        return InspectionOutcome(
            status=status, anomaly_score=det.anomaly_score,
            num_regions=len(det.regions), quality=quality, detection=det.to_dict(),
            regions=[r.to_dict() for r in det.regions],
            decision_log=log.to_list(), iterations=it, escalated=False,
        )
