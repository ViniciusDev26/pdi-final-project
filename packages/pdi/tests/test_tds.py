import numpy as np

from pdi import tds


def _cut(**kw):
    base = dict(
        assimetria=0.15,
        borda_setor=60.0,
        rede=0.004,
        sem_estrutura=0.05,
        pontos=17,
        globulos=6,
        estrias=0,
    )
    return tds.Cutoffs(**(base | kw))


def _lesion(dots: bool):
    img = np.full((300, 300, 3), 200, np.uint8)
    yy, xx = np.mgrid[:300, :300]
    disk = (yy - 150) ** 2 + (xx - 150) ** 2 < 60**2
    img[disk] = (120, 80, 60)
    if dots:
        for y, x in [(130, 130), (150, 170), (170, 140), (140, 160), (160, 125)]:
            img[y - 1 : y + 2, x - 1 : x + 2] = (30, 20, 15)
    return img, disk


def test_score_follows_stolz_weights():
    assert tds.score({"A": 2, "B": 8, "C": 6, "D": 5}) == tds.TDS_MAX
    assert tds.score({"A": 0, "B": 0, "C": 1, "D": 1}) == tds.TDS_MIN
    assert tds.normalized(tds.TDS_MIN) == 0 and tds.normalized(tds.TDS_MAX) == 1


def test_measures_detect_dots_and_sharp_round_border():
    img, mask = _lesion(dots=True)
    m = tds.measures(img, mask)
    assert len(m["borda_setor"]) == 8 and min(m["borda_setor"]) > 0
    assert m["pontos"] >= 4
    assert tds.measures(img, np.zeros_like(mask)) is None


def test_criteria_symmetric_lesion_scores_low():
    img, mask = _lesion(dots=False)
    m = tds.measures(img, mask)
    feats = {"assimetria_eixo_maior": 0.01, "assimetria_eixo_menor": 0.02, "n_cores": 1.0}
    c = tds.criteria(feats, m, _cut(borda_setor=1e9))
    assert c["A"] == 0 and c["B"] == 0 and c["C"] == 1
    assert tds.score(c) < tds.CUTOFF_SUSPICIOUS
    c2 = tds.criteria(
        feats | {"assimetria_eixo_maior": 0.5, "assimetria_eixo_menor": 0.4, "n_cores": 5.0},
        m,
        _cut(borda_setor=0.0),
    )
    assert c2["A"] == 2 and c2["B"] == 8 and tds.score(c2) > tds.CUTOFF_MELANOMA
