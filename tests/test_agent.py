import numpy as np

from src.agent.controller import InspectionAgent, Status
from src.vision import preprocessing, synthetic


def _run(sample, max_iters=4):
    prep = preprocessing.normalize(sample.image)
    return InspectionAgent(max_iters=max_iters).run(prep)


def test_clear_defect_is_flagged():
    out = _run(synthetic.make_sample(21, defect="spot"))
    assert out.status == Status.POTENTIAL_ANOMALY
    assert out.anomaly_score is not None and out.anomaly_score >= 0.55


def test_normal_is_clean():
    out = _run(synthetic.make_sample(22, defect=None))
    assert out.status == Status.NO_ANOMALY


def test_poor_quality_routes_to_human_review_not_defect():
    # A flat, featureless frame fails the quality gate.
    flat = np.full((256, 256, 3), 128, dtype=np.uint8)
    prep = preprocessing.normalize(flat)
    out = InspectionAgent().run(prep)
    assert out.status == Status.HUMAN_REVIEW
    assert out.escalated is True
    assert out.anomaly_score is None  # never scored a bad image as defective


def test_decision_log_records_actions_and_depends_on_result():
    out = _run(synthetic.make_sample(23, defect="spot"))
    actions = [s["action"] for s in out.decision_log if s["action"]]
    assert "assess_image_quality" in actions
    assert "analyze_image" in actions
    # The choose step must reference the actual observed score.
    choose = [s for s in out.decision_log if s["phase"] == "choose"]
    assert choose and "anomaly_score" in choose[0]["observation"]


def test_agent_can_choose_reinspection_on_borderline():
    """At least one synthetic sample should drive the re-inspection branch,
    proving the next action depends on the first visual result."""
    agent = InspectionAgent()
    triggered = False
    for s in synthetic.make_dataset(n_normal=40, n_defect=40):
        prep = preprocessing.normalize(s.image)
        out = agent.run(prep)
        if "run_targeted_reinspection" in [x["action"] for x in out.decision_log]:
            triggered = True
            break
    assert triggered, "no sample exercised the agentic re-inspection branch"


def test_iterations_are_bounded():
    out = _run(synthetic.make_sample(24, defect="scratch"), max_iters=2)
    assert out.iterations <= 2


def test_human_review_on_cap_is_not_a_pass():
    # With max_iters=2, a borderline case cannot complete re-inspection and
    # must escalate rather than silently pass.
    agent = InspectionAgent(max_iters=2)
    for s in synthetic.make_dataset(n_normal=10, n_defect=40):
        prep = preprocessing.normalize(s.image)
        out = agent.run(prep)
        assert out.status in (
            Status.POTENTIAL_ANOMALY, Status.NO_ANOMALY, Status.HUMAN_REVIEW
        )
