# Aberrometria corneana anterior calculada — Tomey TMS (ferramenta de pesquisa)

Programa de computador (Windows; roda também em macOS/Linux) que transforma a exportação numérica
("Data Tables") do topógrafo **Tomey TMS-4 / TMS-4N** em estimativas reprodutíveis da **frente de onda da
superfície anterior da córnea**: mapa de frente de onda, coeficientes de Zernike (OSA), aberração esférica
primária, coma, trefoil e RMS de alta ordem — sempre vinculados à abertura, ao centro, ao modelo óptico e às
versões usadas, com controle de qualidade e proveniência.

Implementa o *Plano de Desenvolvimento — Aberrometria Tomey TMS* (v1.0, 17/10/2026): seções 1–15 e apêndices.

> **Estado: versão de pesquisa 0.1.0.** O perfil de exportação do TMS é **presumido** (manual, seção 3.11.5) e
> ainda **não foi confrontado com arquivos reais** (marco G0/G1). Nada aqui é alegação diagnóstica, seleção de
> LIO ou planejamento terapêutico. Ver `docs/limitacoes.md`.

## O que já está pronto

| Camada (plano) | Estado |
|---|---|
| §3 Adaptador TMS-4/4N (HIT, RAD, ROC, DIO, PUP; Angle/Ring; ausências e interpolados; pareamento; hashes) | Implementado sobre perfil **presumido** `tms4_datatables_v0` — ajustar no G0/G1 |
| §4 Contrato de dados, máscaras, identidade do cálculo por hash | Implementado |
| §5 Reconstrução (cônica + Zernike ordem 10, ou RBF/spline), traçado de raios, OPD, Zernike OSA ordem 6 | Implementado e verificado contra soluções analíticas |
| §6 Núcleo Python + CLI + programa desktop (Qt) + empacotamento Windows | Implementado |
| §7 Interface e relatório (JSON, CSV, PDF derivados do mesmo objeto; três estados por perfil) | Implementado |
| §8 Suíte de verificação (53 testes: léxico, Angle/Ring, pareamento, sinais, orientação, geometria, Zernike, traçado, convergência, lacunas, saídas) | Implementado — `pytest` |
| §9 Estatística do piloto (repetibilidade, Bland–Altman com observações repetidas) + protocolo | Implementado (`tmsab stats`) |
| §0 Inventário da estação / G0 | **Depende de você** — `docs/G0_checklist.md` |
| §9 Piloto com participantes e comparador | **Depende de dados reais** — `docs/protocolo_piloto.md` |

## Instalação (Windows)

**Opção A — executável pronto:** baixe `AberrometriaTMS-windows.zip` (ou `AberrometriaTMS-0.1.0-setup.exe`) gerado
pelo GitHub Actions (aba *Actions* → *Build Windows* → *Artifacts*, ou em *Releases*). Descompacte e abra
`AberrometriaTMS.exe`. A linha de comando fica em `tmsab.exe` na mesma pasta.

**Opção B — compilar no seu PC:** instale Python 3.11+ (python.org, marcando "Add to PATH"), abra o
Prompt de Comando na pasta do projeto e rode `packaging\build_windows.bat`. O resultado fica em
`dist\AberrometriaTMS\`.

**Opção C — rodar do código-fonte (qualquer sistema):**

```bash
pip install -e ".[gui,dev]"
tmsab-gui                 # programa gráfico
tmsab --help              # linha de comando
pytest                    # suíte de verificação
```

## Uso em 5 passos

1. No software do TMS, exporte as **Data Tables** do exame (HIT, RAD, ROC, DIO e, se existir, PUP) para uma
   pasta — **uma pasta por exame e por olho**, com ≥ 4 casas decimais (ver `docs/G0_checklist.md`).
2. Crie o manifesto do pacote (sem dados nominais): `tmsab manifest PASTA --eye OD --model TMS-4N --software X.Y --date 2026-10-20 --capture 1 --decimals 4`.
3. Abra o programa, **Abrir pacote…**, confira a aba *Importação* (estado, diagnósticos, avisos) e *Geometria*
   (altura, cobertura, R, Q, resíduo).
4. Escolha o perfil (centro vértice/pupila, 4 e 6 mm, ordem 6) e **Calcular**. Cada perfil termina em um de três
   estados: **disponível**, **indisponível** (motivo explícito) ou **inválida**.
5. **Exportar relatório** (JSON canônico + CSV + PDF). Para vários exames: `tmsab batch PASTA_RAIZ --out SAIDA`
   gera também `resumo_longo.csv`, entrada direta de `tmsab stats`.

Sem arquivos reais, use **Gerar sintéticos…** (ou `tmsab synth --out PASTA`): 13 casos identificados como
sintéticos (esfera, cônica, aplanático, tórica OD/OS, coma, trefoil, pupila deslocada, lacunas, ruído,
precisão reduzida, Ring/Angle, irregular).

## Verificação científica

```bash
tmsab verify          # oval de Descartes (W≡0), esfera vs Seidel, traçado 3D vs referência meridional, sintéticos
pytest -q             # 53 ensaios (matriz da §8 em docs/validacao.md)
```

Resultados registrados: esfera R=7,8 mm → c(4,0)=+0,0204 µm a 3 mm (Seidel +0,0197); elipsoide Q=−1/n² →
|W| < 10⁻⁵ µm; recuperação de SA/coma/trefoil sintéticos com erro < 0,002 µm (exportação com 6 casas).
**Achado relevante para o G0:** HIT com 3 casas decimais (1 µm) produz viés de −0,035 µm na SA a 6 mm por
quantização sistemática por anel; com 4 casas o erro cai para 0,004 µm (`docs/validacao.md`).

## Estrutura

```
tmsab/            núcleo: model, adapters/, geometry, optics, zernike, quality, pipeline, reporting, plots, stats, synthetic, cli
tmsab/gui/        programa desktop (PySide6)
tmsab/reference/  traçado meridional independente (verificação)
tests/            suíte pytest (§8)
docs/             G0_checklist, dicionario_campos, convencoes, validacao, protocolo_piloto, decisoes, limitacoes
packaging/        PyInstaller spec, build_windows.bat, installer.iss (Inno Setup), build_linux.sh
.github/          workflow que compila o .exe no Windows
```

## Próximo passo concreto

Fechar o **G0**: confirmar modelo/versão do TMS, exportar um exame OD e um OS (Angle e Ring, 4+ casas),
enviar a captura de tela da configuração de exportação e, se possível, Pentacam/OPD do mesmo olho no mesmo dia.
Com isso o perfil `tms4_datatables_v0` é ajustado e marcado como `verificado`.
