from layout_rag import BM25Index, MemoryStore, Retriever
from layout_rag.embed import HashEmbedder
from layout_rag.retrieve import rrf


def _index(result):
    emb = HashEmbedder()
    store = MemoryStore()
    store.add(result.chunks, emb.embed_passages([c.text for c in result.chunks]))
    return store, emb


def test_hybrid_finds_the_right_section(report_result):
    store, emb = _index(report_result)
    hits = Retriever(store, emb).search("cold-chain facility in the northern region", k=3, mode="hybrid")
    assert hits[0].chunk.heading_path[-1] == "2.1 Regional results"
    assert set(hits[0].ranks) <= {"dense", "bm25"}


def test_bm25_and_filters(report_result):
    store, emb = _index(report_result)
    r = Retriever(store, emb, bm25=BM25Index(store.all_chunks()))
    assert r.search("pallets shipped per operating hour", k=1, mode="bm25")[0].chunk.heading_path[-1] == \
        "1.2 Definitions"
    assert r.search("anything", k=5, mode="dense", filters={"doc_name": "nope.pdf"}) == []


def test_rrf_rewards_agreement(report_result):
    a, b, c = report_result.chunks[:3]
    fused = rrf({"x": [a, b, c], "y": [b, a, c]})
    assert {fused[0].chunk.chunk_id, fused[1].chunk.chunk_id} == {a.chunk_id, b.chunk_id}
    assert fused[-1].chunk.chunk_id == c.chunk_id


def test_memory_store_roundtrip(report_result, tmp_path):
    store, emb = _index(report_result)
    store.save(tmp_path)
    again = MemoryStore.load(tmp_path)
    q = emb.embed_queries(["two cross-docking hubs"])[0]
    assert [c.chunk_id for c, _ in again.search(q, 3)] == [c.chunk_id for c, _ in store.search(q, 3)]


def test_reranker_is_applied(report_result):
    class FakeReranker:
        def score(self, query, texts):
            return [1.0 if "Outlook" in t else 0.0 for t in texts]

    store, emb = _index(report_result)
    hits = Retriever(store, emb, reranker=FakeReranker()).search("revenue", k=1, mode="hybrid")
    assert hits[0].chunk.heading_path[-1] == "3. Outlook"
    assert "fused" in hits[0].ranks
