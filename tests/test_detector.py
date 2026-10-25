import numpy as np

from src.vision import detector, preprocessing, synthetic


def _score(sample):
    prep = preprocessing.normalize(sample.image)
    return detector.detect(prep.gray, scale=prep.scale)


def test_defect_scores_higher_than_normal():
    normal = _score(synthetic.make_sample(10, defect=None))
    scratch = _score(synthetic.make_sample(11, defect="scratch"))
    spot = _score(synthetic.make_sample(12, defect="spot"))
    assert scratch.anomaly_score > normal.anomaly_score
    assert spot.anomaly_score > normal.anomaly_score


def test_defect_produces_region():
    res = _score(synthetic.make_sample(13, defect="spot"))
    assert len(res.regions) >= 1
    x, y, w, h = res.regions[0].bbox
    assert w > 0 and h > 0


def test_anomaly_map_is_normalized():
    res = _score(synthetic.make_sample(14, defect="dent"))
    assert res.anomaly_map.min() >= 0.0
    assert res.anomaly_map.max() <= 1.0


def test_score_bounded():
    for seed, kind in [(1, None), (2, "scratch"), (3, "spot"), (4, "dent")]:
        res = _score(synthetic.make_sample(seed, defect=kind))
        assert 0.0 <= res.anomaly_score <= 1.0


def test_lower_threshold_flags_at_least_as_much():
    s = synthetic.make_sample(15, defect="scratch")
    prep = preprocessing.normalize(s.image)
    strict = detector.detect(prep.gray, scale=prep.scale, thresh_k=detector.THRESH_K)
    relaxed = detector.detect(prep.gray, scale=prep.scale, thresh_k=2.0)
    assert relaxed.threshold <= strict.threshold


def test_annotate_returns_same_shape():
    s = synthetic.make_sample(16, defect="spot")
    prep = preprocessing.normalize(s.image)
    res = detector.detect(prep.gray, scale=prep.scale)
    out = detector.annotate(s.image, res)
    assert out.shape == s.image.shape


def test_flat_image_no_regions():
    flat = np.full((256, 256, 3), 128, dtype=np.uint8)
    prep = preprocessing.normalize(flat)
    res = detector.detect(prep.gray, scale=prep.scale)
    assert len(res.regions) == 0
    assert res.anomaly_score < 0.4
