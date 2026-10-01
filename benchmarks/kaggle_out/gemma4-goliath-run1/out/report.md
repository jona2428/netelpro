# Reporte Gemma 4 E2B x Goliath Killer

| Brazo | sl_ood pass@1 | sl_ood pass@8 | sl_train pass@1 | py_ood pass@1 | chars/respuesta (sl_ood) |
|---|---|---|---|---|---|
| A base | 14.4% | 20.0% | 25.7% | 100.0% | 254 |
| B SFT sin filtrar | 15.0% | 15.0% | 27.0% | 100.0% | 316 |
| C SFT verificado | 23.1% | 30.0% | 33.9% | 100.0% | 294 |

D (base + compuerta) = sl_ood pass@8 de A: 20%
E (C + compuerta) = sl_ood pass@8 de C: 30%

Datos: {'ejemplos': 17, 'tareas': 13}. Puerta de memorización pasada: True ([('gate_L1', {'config': {'r': 16, 'epochs': 3, 'lr': 0.0002}, 'pass@8_en_tareas_entrenadas': 0.9230769230769231})])

C - A (sl_ood pass@1): +8.8%  IC95 [+0.0%, +21.9%]
C - B (sl_ood pass@1): +8.1%  IC95 [-3.8%, +23.1%]

**Decisión:** MEJORA PARCIAL (5 <= C-A < 15 pts).  (una semilla; diferencias < 5 pts = empate)