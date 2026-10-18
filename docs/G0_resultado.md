# G0 — resultado da verificação com arquivos reais (18/10/2026)

Estação: PC da clínica (Windows 11), software **TmsSw** (Tomey TMS), cone de **25 anéis**, 256 pontos por anel.
Pacote: 1 paciente, 4 exames do mesmo dia (OD ×2, OS ×2, capturas a ~5 s de intervalo), exportado por
*Ferramentas → Tabelas de Dados* para uma pasta do OneDrive. Transferência para o ambiente de análise por
AnyDesk (área de transferência, zip + base64, SHA-256 conferido). **Nenhum dado nominal** entrou no projeto:
o XREF foi lido apenas nas colunas sem identificação.

## 1. Onde ficam os exames brutos

`C:\Tomey\Tms\NewExams\<número>.tms` — um arquivo binário por exame (ex.: `53703.tms`), listado na coluna
*Filename* do `XREF.dat`. O programa **não** lê o formato nativo; usa a exportação Data Tables.

## 2. Formato real da exportação (perfil `tms4_datatables_v1`)

| Arquivo | Conteúdo | Observado |
|---|---|---|
| `HIT_1_<exame>.dat` | altura sagital (mm) | 256 linhas (ângulos) × **34 colunas**; colunas 1–25 = anéis, 26–34 = zeros; `%9.4f`, vírgula; **negativo = interpolado**, **0 = ausente** |
| `RAD_1_<exame>.dat` | posição radial (mm) | idem; anel 1 a **0,23 mm**, anel 25 a ≈4,4 mm |
| `DIO_1_<exame>.dat` | potência axial (D) | idem |
| `POW_1_<exame>.dat` | potência refrativa (D) | idem (cresce para a periferia, como esperado) |
| `ROC_1_<exame>.dat` | potência tangencial (D) | idem — **em dioptrias, não em mm**, apesar do nome |
| `RNG_1_<exame>.dat` | posição do anel no vídeo (px) | idem |
| `PUP_1_<exame>.dat` | raio da borda pupilar (mm) por ângulo | 256 linhas; negativo = interpolado |
| `AVE_1_<exame>.dat` | potência média e diâmetro médio por anel | 34 linhas × 2 colunas |
| `XREF.dat` | índice: nome, data/hora, olho, caminho nativo, SimK, índices de Klyce, Fourier, excentricidade | CSV com linha de grupos + cabeçalho |

Precisão: 4 casas decimais em mm (0,1 µm) — atende ao requisito do orçamento de erro.
Orientação das tabelas: **Angle** (linhas por ângulo). O programa detecta o perfil pelo nome dos arquivos.

## 3. Ensaios do G0/G1 com os 4 exames

| Ensaio | Resultado |
|---|---|
| Leitura lexical | 4/4 importados; 0 tokens inválidos; 6.188–6.330 pontos medidos, 70–212 interpolados (anel externo, pálpebra), 0 ausentes |
| Dimensões/preenchimento | 34 colunas → 25 anéis após remover as colunas zero; raio médio por anel crescente |
| Faixa física | RAD 0,22–4,52 mm; HIT 0,003–1,38 mm |
| Reconstrução (cônica + Zernike 10) | R 7,55–7,59 mm (K 44,4–44,7 D), Q −0,11 a −0,15; resíduo RMS 0,29–0,36 µm, máx 2–3 µm |
| Pupila (ajuste de círculo às 256 bordas) | diâmetro 6,0 (OD) / 6,4 mm (OS); centro (+0,07, −0,21) OD, (+0,13, −0,15) OS, relativo ao vértice |
| **Orientação angular** (eixo do astigmatismo) | OD: meridiano íngreme calculado **57°** vs Fourier 6 mm do TMS **57,6°** (SimK 35–38°); OS: **157°** vs Fourier 3 mm **159,9°** (SimK 142–155°). Espelhamento ou rotação de 90° dariam 123° e 147°/237° — descartados |
| Magnitude do astigmatismo | 0,50 D refrativo (n 1,376) ≡ 0,44 D ceratométrico × 1,114 — confere com CYL do TMS (0,39–0,44 D) |
| SA c(4,0) 6 mm | OD +0,361 / +0,378 µm; OS +0,435 / +0,445 µm (capturas 1/2) |
| Coma 6 mm | OD 0,29 / 0,14 µm; OS 0,13 / 0,22 µm — **varia entre capturas** (alinhamento/filme lacrimal) |
| Trefoil 6 mm | OD 0,44 / 0,46; OS 0,27 / 0,28 µm — estável |
| RMS HOA 6 mm | 0,57–0,65 µm |

Conclusão: **G0 e G1 fechados** para o perfil `tms4_datatables_v1` (status `verificado_estrutura`). A
convenção angular (0° às 3 horas, anti-horário na vista do observador, sem espelhamento) está confirmada pelo
eixo do astigmatismo em ambos os olhos; um caso com astigmatismo ≥ 1,5 D reforçaria a evidência (o deste
paciente é de 0,3–0,4 D, e o próprio TMS varia 13–22° entre SimK e Fourier nesse nível).

## 4. Pendências

- Confirmar modelo/versão do software na tela "Sobre" do TmsSw (o formato indica TMS-4/4N com cone de 25).
- Olho de calibração (esfera de raio conhecido), se existir no fluxo da Tomey — teste de unidade mais limpo.
- Comparador (Pentacam/OPD) do mesmo olho para o G3.
- Repetibilidade: as 2 capturas por olho já mostram SA estável (Δ ≤ 0,02 µm) e coma variável (Δ até 0,15 µm);
  o piloto (§9) quantifica com 3 capturas e 20–30 participantes.
