"""Social graph communities (/api/social-graph/communities): input handling and modularity."""


def test_communities_two_triangles(client):
    r = client.post("/api/social-graph/communities", json={"edges": [[1, 2], ["2", 3], [3, 1], [4, 5], [5, 6], [6, 4], [3, 4]]}).json()
    # Two triangles joined by one edge: Q = 1/2 - 1/7 ... computed by hand = 5/14
    assert r["n_communities"] == 2 and abs(r["modularity"] - 5 / 14) < 1e-12
    assert sorted(len(c["members"]) for c in r["communities"]) == [3, 3]


def test_bad_weight_is_422(client):
    assert client.post("/api/social-graph/communities", json={"edges": [["a", "b", "x"], ["b", "c"]]}).status_code == 422
