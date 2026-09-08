# Ideas de aplicación — anotadas 2026-09-07

Conversación con Jona sobre hacia dónde llevar Netelpro más allá de Neuromancer.
De cinco ideas propuestas, estas dos quedaron elegidas para explorar primero.

## 1. Motor de reglas genérico para aprobación de acciones sensibles

El patrón que ya corre en producción dentro de Neuromancer (`netelpro_gate.sl`,
`verification_rule.sl`, `shutdown_rule.sl`): el host (LLM o humano) juzga
intención difusa, la regla compilada aplica la consecuencia de forma
determinista y verificable exhaustivamente (test diferencial contra el
intérprete, tabla de verdad completa cuando el espacio de entradas lo permite).

Aplicaciones fuera de Neuromancer: aprobación de transacciones/gastos, gates
de moderación de contenido, interlocks de seguridad en robótica, motores de
compliance (KYC/AML, elegibilidad de crédito) — cualquier lugar donde hoy hay
un `if/else` disperso en Python decidiendo algo crítico sin poder probarse
exhaustivamente.

**Trabajo necesario:** separar el patrón de gate de Neuromancer como librería
standalone, documentada para consumo externo (hoy vive acoplado a
`zone_policy.py` de Neuromancer).

**Riesgo declarado:** compite con motores de policy ya establecidos (OPA/Rego).
La diferencia real a vender no es "otro motor de reglas" — es "diseñado para
que un LLM lo escriba y se auto-verifique sin intervención humana en el loop".

## 5. Infraestructura de gates para otros frameworks de agentes

No limitar el patrón de gate compilado a Neuromancer — ofrecerlo como pieza
que cualquier harness de agentes (LangChain, AutoGPT-style, otros) pueda
embeber para sus propias aprobaciones de tool-calling, sin reimplementar la
lógica de verificación cada vez.

**Trabajo necesario:** definir un contrato de integración genérico (qué tipos
de entrada acepta un gate, cómo se compila, cómo se consulta en runtime) que
no asuma la arquitectura específica de Neuromancer.

---

## Dirección elegida para seguir primero: RLVR (verifier-guided training)

Idea que surgió de la propia conversación, no de la lista original: usar el
parser fiscal de Netelpro (¿compiló?) y las reglas `.sl` (¿pasó la tabla de
verdad?) como señal de recompensa para entrenamiento por refuerzo — mismo
principio que usan los modelos de matemática/código (DeepSeek-R1 y similares):
la recompensa viene de un chequeo mecánico real, no de preferencia humana
etiquetada a mano.

Por qué generaliza mejor que el DPO actual (evidencia propia, no teórica): el
split OOD de hoy (`benchmarks/vtb_ood_dataset.py` / `vtb_ood_runner.py`) mostró
que 106 ejemplos DPO etiquetados a mano solo enseñan la plantilla angosta que
alcanzaron a cubrir — un system prompt de 3 líneas reprodujo la mayor parte
del efecto sin entrenar nada. Una señal de verificador real (compiló o no,
pasó la regla o no) enseña el invariante, no la superficie de la plantilla.

**Punto de partida propuesto:** el dominio más angosto donde el verificador YA
existe — generación de reglas `.sl` correctas. No hace falta construir un
verificador nuevo, solo conectar la señal de "compiló + pasó el diferencial"
como recompensa.

**Estado:** sin planificar todavía. Próxima sesión.
