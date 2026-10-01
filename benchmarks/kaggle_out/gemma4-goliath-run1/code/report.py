"""Lee out/eval_*.json y escribe out/report.md con la decisión según los criterios del README."""
import json, random
from config import OUT, K


def load(n):
    f = OUT / f"eval_{n}.json"
    return json.loads(f.read_text()) if f.exists() else None


def boot(a, b, key="sl_ood", n=4000, seed=0):
    """Diferencia media de pass@1 por tarea (b - a) con IC95 bootstrap sobre tareas."""
    ta, tb = a[key]["per_task"], b[key]["per_task"]
    ids = [t for t in ta if t in tb]
    d = [(tb[t] - ta[t]) / K for t in ids]
    rng = random.Random(seed)
    ms = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), ms[int(.025 * n)], ms[int(.975 * n)]


A, B, C = load("base"), load("raft_unverified"), load("raft_verified")
st = json.loads((OUT / "status.json").read_text()) if (OUT / "status.json").exists() else {}
L = ["# Reporte Gemma 4 E2B x Goliath Killer\n"]
L.append("| Brazo | sl_ood pass@1 | sl_ood pass@8 | sl_train pass@1 | py_ood pass@1 | chars/respuesta (sl_ood) |\n|---|---|---|---|---|---|")
for name, r in (("A base", A), ("B SFT sin filtrar", B), ("C SFT verificado", C)):
    if r:
        f = lambda k, m: f"{r[k][m]:.1%}" if k in r else "-"
        L.append(f"| {name} | {f('sl_ood','pass@1_mean')} | {f('sl_ood','pass@k')} | {f('sl_train','pass@1_mean')} | {f('py_ood','pass@1_mean')} | {r['sl_ood'].get('mean_chars','-') if 'sl_ood' in r else '-'} |")
L.append(f"\nD (base + compuerta) = sl_ood pass@8 de A: {A['sl_ood']['pass@k']:.0%}" if A else "")
L.append(f"E (C + compuerta) = sl_ood pass@8 de C: {C['sl_ood']['pass@k']:.0%}" if C else "")
L.append(f"\nDatos: {st.get('verified')}. Puerta de memorización pasada: {st.get('gate_passed')} "
         f"({[ (k, v) for k, v in st.items() if k.startswith('gate_L')]})")
if A and C:
    d, lo, hi = boot(A, C); L.append(f"\nC - A (sl_ood pass@1): {d:+.1%}  IC95 [{lo:+.1%}, {hi:+.1%}]")
    dcb = boot(B, C) if B else None
    if dcb: L.append(f"C - B (sl_ood pass@1): {dcb[0]:+.1%}  IC95 [{dcb[1]:+.1%}, {dcb[2]:+.1%}]")
    py = C["py_ood"]["pass@1_mean"] - A["py_ood"]["pass@1_mean"] if "py_ood" in C and "py_ood" in A else None
    v = []
    if not st.get("gate_passed"): v.append("PUERTA DE MEMORIZACIÓN NO PASADA: el LoRA no inyectó la gramática; OOD no es interpretable")
    if d >= .15 and dcb and dcb[0] >= .05: v.append("EL MÉTODO FUNCIONA (C-A >= 15 pts y C-B >= 5 pts)")
    elif d >= .05 and dcb and dcb[0] < .05: v.append("SOLO ERA VOLUMEN (C-B < 5 pts): mejora pero no por el verificador")
    elif d < .05: v.append("SIN SEÑAL (C-A < 5 pts)")
    else: v.append("MEJORA PARCIAL (5 <= C-A < 15 pts)")
    if py is not None and py < -.05: v.append(f"DAÑO COLATERAL en Python ({py:+.1%})")
    L.append("\n**Decisión:** " + "; ".join(v) + ".  (una semilla; diferencias < 5 pts = empate)")
(OUT / "report.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))
