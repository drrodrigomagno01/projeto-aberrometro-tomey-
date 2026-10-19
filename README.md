# Aberrometria corneana anterior calculada — Tomey TMS (ferramenta de pesquisa)

Programa de computador (Windows; roda também em macOS/Linux) que transforma a exportação numérica
("Data Tables") do topógrafo **Tomey TMS-4 / TMS-4N** em estimativas reprodutíveis da **frente de onda da
superfície anterior da córnea**: mapa de frente de onda, coeficientes de Zernike (OSA), aberração esférica
primária, coma, trefoil e RMS de alta ordem; e, desde a 0.4.0, a **ceratometria por zona** (K1/K2 e eixos por
diâmetro, da potência axial medida pelo TMS) e o **chord mu aparente** — sempre vinculados à abertura, ao centro, ao
modelo óptico e às versões usadas, com controle de qualidade e proveniência.

Implementa o *Plano de Desenvolvimento — Aberrometria Tomey TMS* (v1.0, 17/10/2026): seções 1–15 e apêndices.

> **Estado: versão de pesquisa 0.4.1.** O formato real das Data Tables do TMS-4N foi verificado com exames
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
| §7 Interface e relatório (JSON, CSV, PDF derivados do mesmo objeto; três estados por perfil) + **página "Resultado completo"** (topografia axial/tangencial/refrativa/elevação + frente de onda + Zernike + tabelas, 1 folha A4) | Implementado |
| **Ceratometria por zona** (0.4.0): K1 plano, K2 curvo e eixos em Ø 1–8 mm (e 2,4/3,3 mm), zonas 0–3 e 0–6 mm, zona pupilar e os 25 anéis, da potência axial medida pelo TMS; SimK e ACP do TMS recalculados e conferidos a cada exame; alerta de astigmatismo dependente da zona; **controle de diâmetro** (0.4.1): no programa e no site, um controle deslizante ou um clique no mapa axial escolhe o diâmetro (0,5 mm até o último anel, passo de 0,1 mm) e mostra o círculo, os meridianos K1/K2 e K1/K2/eixos/ΔK/Km da circunferência Ø d e da zona 0–d | Implementado e **conferido com 10 exames reais** (SimK a ≤ 0,01 D em 9 de 10 exames e 0,03 D no 10º (27204); Km da zona pupilar × ACP: erro médio 0,0045 D (máx. 0,013) nos 9 com pupila detectada) — `docs/convencoes.md`, D23, D26, D27 |
| **Topografia no estilo do TMS** (0.4.0): escala Standard 40,5–53,0 D (paleta do aparelho), transferidor, meridianos íngreme/plano com as potências a 3/5/7 mm (*Astig Stats*), pupila; **botões de visualização** (mapa, escala, zona, diâmetro pupilar 3/4/5/6 mm ou medido, centro) no programa e no HTML | Implementado — D24/D25 |
| §8 Suíte de verificação (230 testes: léxico, Angle/Ring, pareamento, sinais, orientação, geometria, Zernike, traçado, convergência, lacunas, ceratometria, topografia, saídas) | Implementado — `pytest` |
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
são calculados um a um (~15 s cada) → painel **Topografia (estilo TMS)** com os mesmos botões do programa (mapa,
escala, zona, diâmetro pupilar, centro) e *Aplicar estes filtros à página de resultado completo* → **Imprimir
resultado completo** (1 página A4 por exame) ou *Imprimir relatório detalhado*, *Salvar relatório (.html)*, ou
baixar JSON/CSV/PDF por exame. Na seção de ceratometria de cada exame fica o mesmo **controle de diâmetro** do
programa (controle deslizante ou clique no mapa axial; funciona também no relatório .html salvo). Os números são
idênticos aos do programa de computador (mesmo código;
verificado a 12 casas decimais). Pastas com subpastas (uma por exportação) e pastas com dois pacientes são tratadas
como no programa.

**Publicada em https://aberrometria-tms.vercel.app** (Vercel, projeto `aberrometria-tms`): a mesma página, servida por
HTTPS. O servidor só entrega o arquivo; a pasta do TMS continua sendo lida e calculada no navegador. Para publicar uma
versão nova, depois que o CI do commit passar: `web/publish_vercel.sh` (baixa o HTML gerado pelo CI, envia só a página
e confere o sha256 servido).

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
3. A aba **Resultado completo** mostra, numa folha A4, o mapa de potência axial **no estilo do TMS** (escala
   Standard 40,5–53,0 D em passos de 0,5 D com a paleta do aparelho, transferidor, meridianos íngreme/plano com as
   potências a 3/5/7 mm, caixa *Astig Stats*, pupila), mais tangencial, refrativa, elevação, frente de onda total e
   de alta ordem, Zernike, geometria/pupila (e chord mu aparente)/astigmatismo, a **ceratometria por zona**, a tabela
   dos perfis (3/4/5/6 mm e pupila medida, centro no vértice e na pupila) e os índices do TMS —
   **Imprimir resultado completo…** manda direto para a impressora; a mesma página é a 1ª do PDF em `analises/`.
4. A aba **Topografia (TMS)** tem os **botões de visualização**: mapa (axial, tangencial, refrativa, elevação, frente
   de onda, alta ordem), escala (Standard do TMS ou ajustada ao exame), zona (tudo, 3, 5, 7 mm), **diâmetro pupilar**
   (medido pelo TMS, 3, 4, 5, 6 mm) e centro (vértice/pupila). Os filtros valem também para a página de resultado
   completo e para a impressão. As demais abas detalham: *Ceratometria* (K1 plano, K2 curvo e eixos em todos os diâmetros, zonas 0–3/0–6 mm,
   pupila, SimK do TMS e os 25 anéis, com *Copiar tabela (TSV)*; à direita, o **mapa axial com o controle de
   diâmetro** — arraste o controle, use as setas ou clique/arraste no mapa: o círculo Ø d, os meridianos K1 (azul) e
   K2 (laranja) e os números da circunferência e da zona 0–d naquele diâmetro — e os gráficos por diâmetro), *Importação* (diagnóstico,
   SimK/índices do TMS), *Geometria*, *Frente de onda*, *Zernike*, *Comparação* (ex.: as duas capturas do mesmo olho),
   *Técnico*.

Linha de comando: `tmsab analyze "D:\TMS\<paciente>"` faz o mesmo (relatórios em `analises/`);
`tmsab batch RAIZ --out SAIDA` processa várias pastas de pacientes e gera `resumo_longo.csv` para `tmsab stats`.

Nome do paciente: a página **Resultado completo** (aba do programa, 1ª página do PDF, figura do HTML) traz **nome
do paciente e data do exame** lidos do `XREF.dat` — é o laudo para o prontuário; a opção *"Nome do paciente (XREF) e
data no resultado completo"* pode ser desligada. JSON, CSV e as páginas técnicas do PDF **nunca** levam o nome: usam o
código de estudo (`P-XXXXXXXX`, hash do nome/ID calculado na leitura e descartado, igual para o mesmo paciente em
qualquer pasta); `tmsab audit PASTA` confere isso (nome permitido só na 1ª página dos PDFs). **Atenção:** exportar de
novo para uma pasta que já tem uma exportação faz o TMS *acrescentar* os exames ao `XREF.dat` — o programa separa os
pacientes pelo código e avisa, mas o certo é uma pasta nova por paciente. Uma pasta com subpastas (uma por
exportação) também pode ser aberta de uma vez.

Sem arquivos reais, use **Gerar sintéticos…** (ou `tmsab synth --out PASTA`): 13 casos identificados como
sintéticos (perfil v0) — e `tmsab.synthetic.write_export_folder_v1` gera pastas no formato real para testes.

## Verificação científica

```bash
tmsab verify          # oval de Descartes (W≡0), esfera vs Seidel, traçado 3D vs referência meridional, sintéticos
pytest -q             # 230 ensaios (matriz da §8 em docs/validacao.md + formato real + ceratometria + topografia)
```

Resultados registrados: esfera R=7,8 mm → c(4,0)=+0,0204 µm a 3 mm (Seidel +0,0197); elipsoide Q=−1/n² →
|W| < 10⁻⁵ µm; recuperação de SA/coma/trefoil sintéticos com erro < 0,002 µm (exportação com 6 casas).
**Achado relevante para o G0:** HIT com 3 casas decimais (1 µm) produz viés de −0,035 µm na SA a 6 mm por
quantização sistemática por anel; com 4 casas o erro cai para 0,004 µm (`docs/validacao.md`).

## Estrutura

```
tmsab/            núcleo: model, adapters/, geometry, optics, zernike, quality, keratometry, pupila, pipeline, reporting, plots, stats, synthetic, cli
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
