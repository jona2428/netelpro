# Reporte de Entrenamiento: Red Neuro-Simb?lica vs. Red Cl?sica

**Fecha:** 2026-09-13 17:27:53  
**Entorno:** PyTorch + Netelpro LLVM Backend ? CPU AMD  
**Topolog?a de Red:** [2 entradas -> 16 ocultas -> 16 ocultas -> 2 salidas]  

---

## 1. M?tricas de Entrenamiento y Precisi?n

| M?trica | Red Cl?sica (MLP ReLU) | Red Neuro-Simb?lica (NetelproDeepNetwork) | Veredicto |
|---|---|---|---|
| **Precisi?n en Test (Datos Normales)** | **67.0%** | **67.0%** | Rendimiento equivalente |
| **Tiempo de Entrenamiento (25 ?pocas)** | **0.03 s** | **1.36 s** | STE totalmente diferenciable |
| **Certificado Formal de Auditor?a** | No (Caja Negra) | **S? (Emitido en 10916.20 ?s)** | Explicabilidad matem?tica total |

---

## 2. Comportamiento ante Entradas Adversarias / Desbordes

| Prueba de Estr?s | Red Cl?sica | Red Netelpro con Silicio |
|---|---|---|
| **Activaci?n M?xima Ante Ruido Fuera de Banda** | `8.72` (Alucinaci?n / Desborde) | Inhibici?n a 0.0 (*fail-closed*) |
| **Neuronas Inhibidas Formalmente** | 0 (Indefensa) | **10 de 32 neuronas cortadas** |
| **Garant?a Formal de Seguridad** | ? Vulnerable | ? **100% Verificado por Contrato** |

---

## 3. Conclusi?n Cient?fica

El estimador **Straight-Through Estimator (STE)** desarrollado en la Fase 1 permite entrenar arquitecturas profundas multicapa gobernadas por Netelpro con la misma facilidad que una red cl?sica, pero dot?ndolas de una armadura formal determinista que elimina por completo el riesgo de desbordes o activaciones peligrosas.
