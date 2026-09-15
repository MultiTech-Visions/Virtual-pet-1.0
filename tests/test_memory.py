import numpy as np

from festival_pet.memory import ENCOUNTER_GAP, MAX_EMBEDDINGS, FaceMemory


def _vec(seed, dim=128):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def test_enroll_match_and_persist(tmp_path):
    path = tmp_path / "mem.json"
    mem = FaceMemory(path, save_interval=0.0)
    a = _vec(1)
    assert mem.match(a) == (None, -1.0)
    p = mem.enroll(a, now=100.0)
    hit, sim = mem.match(a + 0.05 * _vec(9))
    assert hit is p and sim > 0.9
    miss, _ = mem.match(_vec(2))
    assert miss is None
    mem.save(force=True)

    again = FaceMemory(path)
    assert again.people[p.person_id].encounters == 1
    hit2, _ = again.match(a)
    assert hit2.person_id == p.person_id


def test_reinforce_caps_embeddings():
    mem = FaceMemory("/nonexistent/never-written.json")
    p = mem.enroll(_vec(1), now=0.0)
    for i in range(20):
        mem.reinforce(p, _vec(100 + i), similarity=0.5)
    assert len(p.embeddings) == MAX_EMBEDDINGS
    mem.reinforce(p, _vec(1), similarity=0.95)  # near-duplicate adds nothing
    assert len(p.embeddings) == MAX_EMBEDDINGS


def test_relationship_tiers_progress():
    mem = FaceMemory("/nonexistent/never-written.json")
    p = mem.enroll(_vec(1), now=0.0)
    assert p.tier() == "acquaintance"
    assert mem.sighted(p, now=10.0) is False  # same visit
    assert mem.sighted(p, now=10.0 + ENCOUNTER_GAP + 1) is True
    assert p.tier() == "friend"
    for _ in range(3):
        mem.add_hold(p)
        mem.add_pet(p)
    mem.add_attention(p, 120.0)
    assert p.tier() == "bestie"
    assert 0.0 < p.affection <= 1.0
    s = mem.summary()
    assert s["people"] == 1 and s["top"][0]["tier"] == "bestie"
