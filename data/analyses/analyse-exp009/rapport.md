# EXP-009 : analyse pré-enregistrée (GPTQ contre RTN, à mémoire égale)

Contrôles passés, commit 090bdb5.

| Test | Tâche, version | GPTQ moins RTN (points) | p | p Holm | Verdict |
| --- | --- | --- | --- | --- | --- |
| G1 | pusht all:4 | +12.6 [+9.7, +15.7] | 0.0001 | 0.0004 | soutenue |
| G2 | tworoom all:4 | +10.8 [+8.6, +13.1] | 0.0001 | 0.0004 | soutenue |
| G3 | pusht all:3 | +39.5 [+35.8, +43.3] | 0.0001 | 0.0004 | soutenue |
| G4 | tworoom all:3 | +45.2 [+41.4, +49.2] | 0.0001 | 0.0004 | soutenue |

| Tâche, version | fp32 | RTN | GPTQ | GPTQ moins RTN | Part de la perte rattrapée |
| --- | --- | --- | --- | --- | --- |
| pusht all:4 | 83.5 % | 69.4 % | 82.0 % | +12.6 [+9.7, +15.7] | +89.5 [+77.2, +102.7] % |
| pusht all:3 | 83.5 % | 35.7 % | 75.2 % | +39.5 [+35.8, +43.3] | +82.6 [+77.3, +87.9] % |
| pusht encoder:4 | 83.5 % | 78.9 % | 83.1 % | +4.2 [+2.2, +6.2] | +90.5 [+64.4, +123.9] % |
| pusht autres:4 | 83.5 % | 74.1 % | 82.8 % | +8.8 [+6.1, +11.4] | +92.8 [+75.8, +111.8] % |
| tworoom all:4 | 84.8 % | 73.8 % | 84.6 % | +10.8 [+8.6, +13.1] | +97.8 [+89.0, +107.5] % |
| tworoom all:3 | 84.8 % | 38.6 % | 83.9 % | +45.2 [+41.4, +49.2] | +98.0 [+95.5, +100.4] % |
| tworoom encoder:4 | 84.8 % | 75.7 % | 84.9 % | +9.2 [+6.8, +11.6] | +101.3 [+91.8, +111.8] % |
| tworoom autres:4 | 84.8 % | 83.6 % | 84.8 % | +1.2 [+0.1, +2.3] | sans objet |
