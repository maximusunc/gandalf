import json

with open("./imatinib_paths.json", "r", encoding="utf-8") as f:
    gandalf = json.load(f)

gandalf_edges = set()
for path in gandalf:
    gandalf_edges.add(
        (path["e0"]["subject"], path["e0"]["predicate"], path["e0"]["object"])
    )

with open("./CHEBI:45783_response.json", "r", encoding="utf-8") as f:
    retriever = json.load(f)
    print("Retriever returned results:", len(retriever["message"]["results"]))

gandalf_missed_edges = []
retriever_missed_edges = []
retriever_edges = set()
for edge in retriever["message"]["knowledge_graph"]["edges"].values():
    retriever_edge = (edge["subject"], edge["predicate"], edge["object"])
    retriever_edges.add(retriever_edge)
    if retriever_edge not in gandalf_edges:
        gandalf_missed_edges.append(retriever_edge)

for gandalf_edge in gandalf_edges:
    if gandalf_edge not in retriever_edges:
        retriever_missed_edges.append(gandalf_edge)


with open("gandalf_missed_edges.json", "w", encoding="utf-8") as f:
    json.dump(gandalf_missed_edges, f, indent=2)

with open("retriever_missed_edges.json", "w", encoding="utf-8") as f:
    json.dump(retriever_missed_edges, f, indent=2)
