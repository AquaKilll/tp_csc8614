# TP5/test_graph_minimal.py
import uuid

from TP5.load_test_emails import load_all_emails
from TP5.agent.state import AgentState
from TP5.agent.graph_minimal import build_graph

if __name__ == "__main__":
    emails = load_all_emails()
    e = emails[8]

    state = AgentState(
        run_id=str(uuid.uuid4()),
        email_id=e["email_id"],
        subject=e["subject"],
        sender=e["from"],
        body=e["body"],
    )

    app = build_graph()
    out = app.invoke(state)

    print("=== DECISION ===")
    print(out["decision"].model_dump_json(indent=2))
    print("\n=== DRAFT_V1 ===")
    print(out["draft_v1"])   # TODO: afficher draft_v1
    print("\n=== ACTIONS ===")
    print(out["actions"])   # TODO: afficher actions

    app = build_graph()
    out = app.invoke(state)

    print("=== DECISION ===")
    print(out["decision"].model_dump_json(indent=2))
    
    print("\n=== DRAFT_V1 ===")
    print(out["draft_v1"])

    print("\n=== ACTIONS ===")
    print(out["actions"])

    # --- AJOUT EVIDENCE POUR EXERCICE 6 ---
    print("\n=== EVIDENCE (Documents trouvés) ===")
    # On vérifie si la liste existe et n'est pas vide
    if "evidence" in out and out["evidence"]:
        for doc in out["evidence"]:
            print(f"- [{doc.doc_id}] {doc.doc_type} ({doc.source})")
            # print(f"  Extrait: {doc.snippet[:100]}...") # Optionnel
    else:
        print("Aucun document trouvé (ou liste vide).")

    print("\n=== FINAL ===")
    print("kind =", out["final_kind"])
    print("text =", out["final_text"])