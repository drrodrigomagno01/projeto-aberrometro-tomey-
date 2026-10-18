# Aberrometria corneana anterior calculada — Tomey TMS (ferramenta de pesquisa)

Programa de computador (Windows; roda também em macOS/Linux) que transforma a exportação numérica
("Data Tables") do topógrafo **Tomey TMS-4 / TMS-4N** em estimativas reprodutíveis da **frente de onda da
superfície anterior da córnea**: mapa de frente de onda, coeficientes de Zernike (OSA), aberração esférica
primária, coma, trefoil e RMS de alta ordem — sempre vinculados à abertura, ao centro, ao modelo óptico e às
versões usadas, com controle de qualidade e proveniência.

Implementa o *Plano de Desenvolvimento — Aberrometria Tomey TMS* (v1.0, 17/10/2026): seções 1–15 e apêndices.

> **Estado: versão de pesquisa 0.2.2.** O formato real das Data Tables do TMS-4N foi verificado com exames
> reais de 3 pacientes (perfil `tms4_datatables_v1`; orientação angular confirmada com astigmatismo de até 2,5 D;
> marcos G0/G1 fechados — `docs/G0_resultado.md`). Nada aqui é alegação
> diagnóstica, seleção de LIO ou planejamento terapêutico. Ver `docs/limitacoes.md`.

## O que já está pronto

| Camada (plano) | Estado |
|---|---|
| §3 Adaptador TMS-4/4N (HIT, RAD, PUP, AVE, DIO/POW/ROC/RNG; XREF sem nomes; ausências e interpolados; hashes) | Implementado e **verificado com exames reais** (`tms4_datatables_v1`); perfil v0 mantido para pacotes sintéticos |
| §4 Contrato de dados, máscaras, identidade do cálculo por hash | Implementado |
| §5 Reconstrução (cônica + Zernike ordem 10, ou RBF/spline), traçado de raios, OPD, Zernike OSA ordem 6 | Implementado e verificado contra soluções analíticas |
| §6 Núcleo Python + CLI + programa desktop (Qt) + empacotamento Windows/macOS + versão HTML (Pyodide) | Implementado |
| §7 Interface e relatório (JSON, CSV, PDF derivados do mesmo objeto; três estados por perfil) | Implementado |
| §8 Suíte de verificação (66 testes: léxico, Angle/Ring, pareamento, sinais, orientação, geometria, Zernike, traçado, convergência, lacunas, saídas) | Implementado — `pytest` |
| §9 Estatística do piloto (repetibilidade, Bland–Altman com observações repetidas) + protocolo | Implementado (`tmsab stats`) |
| §0 Inventário da estação / G0–G1 | **Fechados em 18/10/2026** com 14 exames reais de 3 pacientes — `docs/G0_resultado.md` (pendentes: tela "Sobre", olho de calibração, comparador) |
| §9 Piloto com participantes e comparador | **Depende de dados reais** — `docs/protocolo_piloto.md` |

## Instalação (Windows)

**Opção A — executável pronto:** baixe `AberrometriaTMS-windows.zip` (ou `AberrometriaTMS-<versão>-setup.exe`) em
[Releases](https://github.com/drrodrigomagno01/projeto-aberrometro-tomey-/releases) (gerado pelo GitHub Actions a cada
versão `v*`; builds intermediários ficam em *Actions → Build → Artifacts*). Descompacte e abra
`AberrometriaTMS.exe`. A linha de comando fica em `tmsab.exe` na mesma pasta. O Windows SmartScreen pode avisar
"editor desconhecido" (executável sem assinatura digital): *Mais informações → Executar assim mesmo*.

**Opção B — compilar no seu PC:** instale Python 3.11+ (python.org, marcando "Add to PATH"), abra o
Prompt de Comando na pasta do projeto e rode `packaging\build_windows.bat`. O resultado fica em
`dist\AberrometriaTMS\`.

## Instalação (macOS)

Baixe `AberrometriaTMS-<versão>-macos-arm64.dmg` (Apple Silicon: M1/M2/M3/M4) em Releases ou em *Actions → Build →
Artifacts*, abra o .dmg e arraste **AberrometriaTMS** para *Aplicativos*. O app não tem assinatura de desenvolvedor
Apple, então na primeira abertura o macOS bloqueia: vá em *Ajustes do Sistema → Privacidade e Segurança* e clique em
**"Abrir mesmo assim"** (ou, no Terminal: `xattr -cr /Applications/AberrometriaTMS.app`). A linha de comando fica em
`/Applications/AberrometriaTMS.app/Contents/MacOS/tmsab`. Macs Intel: use a opção C abaixo ou a versão HTML.

## Versão HTML (qualquer computador, sem instalar nada)

`AberrometriaTMS-<versão>.html` (em Releases / *Actions → Build → Artifacts*, ou gerado por `python web/build_html.py`)
é um **arquivo único** que roda no Chrome, Edge ou Safari: o mesmo núcleo Python (NumPy/SciPy/Matplotlib) executa
dentro do navegador via Pyodide/WebAssembly — **sem servidor, sem banco de dados, nenhum dado sai do computador**.
Na primeira abertura o navegador baixa o motor (~35 MB de cdn.jsdelivr.net; fica em cache). Uso: abrir o .html →
*Escolher pasta exportada do TMS* (o navegador pergunta se pode "carregar" os arquivos: é leitura local) → os exames
são calculados um a um (~10 s cada) → **Imprimir relatório** (⌘P/Ctrl+P → impressora ou PDF), *Salvar relatório
(.html)* ou baixar JSON/CSV/PDF por exame. Os números são idênticos aos do programa de computador (mesmo código;
verificado a 12 casas decimais). Pastas com subpastas (uma por exportação) e pastas com dois pacientes são tratadas
como no programa.

**Opção C — rodar do código-fonte (qualquer sistema):**

```bash
pip install -e ".[gui,dev]"
tmsab-gui                 # programa gráfico
tmsab --help              # linha de comando
pytest                    # suíte de verificação
```

## Uso no dia a dia (3 passos)

1. No TmsSw: selecione o paciente → **Ferramentas → Tabelas de Dados** → *Select All Exams of the current Patient*
   → OK → salve numa **pasta nova por paciente** (ex.: `D:\TMS\<paciente>`). Saem `HIT_1_<exame>.dat`,
   `RAD_1_<exame>.dat`, `PUP_…`, `AVE_…`, `DIO/POW/ROC/RNG_…` e `XREF.dat`.
2. No programa: **Arquivo → Abrir pasta exportada do TMS (paciente)…** e escolha essa pasta. O programa
   identifica cada exame (olho, data/hora e número da captura pelo XREF), **calcula automaticamente** os perfis
   padrão (vértice e pupila, 4 e 6 mm, Zernike ordem 6) e **salva JSON + CSV + PDF em `<pasta>/analises/`**.
   (As duas automações podem ser desligadas nas caixas do painel esquerdo.)
3. Navegue pelas abas: *Importação* (diagnóstico, SimK/índices do TMS), *Geometria*, *Frente de onda*,
   *Zernike*, *Comparação* (ex.: as duas capturas do mesmo olho), *Técnico*.

Linha de comando: `tmsab analyze "D:\TMS\<paciente>"` faz o mesmo (relatórios em `analises/`);
`tmsab batch RAIZ --out SAIDA` processa várias pastas de pacientes e gera `resumo_longo.csv` para `tmsab stats`.

Nome do paciente: pode estar no nome da pasta e está no `XREF.dat`, mas **não entra em nenhum relatório** —
o código de estudo (`P-XXXXXXXX`) é um hash do nome/ID lido do XREF e descartado na hora, igual para o mesmo
paciente em qualquer pasta; o olho, a data/hora e o número da captura identificam o exame. **Atenção:** exportar de
novo para uma pasta que já tem uma exportação faz o TMS *acrescentar* os exames ao `XREF.dat` — o programa separa os
pacientes pelo código e avisa, mas o certo é uma pasta nova por paciente. Uma pasta com subpastas (uma por
exportação) também pode ser aberta de uma vez.

Sem arquivos reais, use **Gerar sintéticos…** (ou `tmsab synth --out PASTA`): 13 casos identificados como
sintéticos (perfil v0) — e `tmsab.synthetic.write_export_folder_v1` gera pastas no formato real para testes.

## Verificação científica

```bash
tmsab verify          # oval de Descartes (W≡0), esfera vs Seidel, traçado 3D vs referência meridional, sintéticos
pytest -q             # 66 ensaios (matriz da §8 em docs/validacao.md + formato real em tests/test_adapter_v1.py)
```

Resultados registrados: esfera R=7,8 mm → c(4,0)=+0,0204 µm a 3 mm (Seidel +0,0197); elipsoide Q=−1/n² →
|W| < 10⁻⁵ µm; recuperação de SA/coma/trefoil sintéticos com erro < 0,002 µm (exportação com 6 casas).
**Achado relevante para o G0:** HIT com 3 casas decimais (1 µm) produz viés de −0,035 µm na SA a 6 mm por
quantização sistemática por anel; com 4 casas o erro cai para 0,004 µm (`docs/validacao.md`).

## Estrutura

```
tmsab/            núcleo: model, adapters/, geometry, optics, zernike, quality, pipeline, reporting, plots, stats, synthetic, cli
tmsab/gui/        programa desktop (PySide6)
tmsab/web.py      versão navegador (chamado pelo Pyodide); web/template.html + web/build_html.py geram o .html único
tmsab/reference/  traçado meridional independente (verificação)
tests/            suíte pytest (§8)
docs/             G0_checklist, G0_resultado, dicionario_campos, convencoes, validacao, protocolo_piloto, decisoes, limitacoes
packaging/        PyInstaller spec, build_windows.bat, installer.iss (Inno Setup), build_linux.sh
.github/          workflow que compila o .exe no Windows
```

## Próximo passo concreto

**G3**: Pentacam ou OPD-Scan do mesmo olho no mesmo dia (3–5 pacientes) para conferir o modelo óptico em olhos
reais; depois o piloto (`docs/protocolo_piloto.md`). Pendências menores do G0 em `docs/G0_resultado.md`.
