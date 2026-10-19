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
eixo do astigmatismo em ambos os olhos; o ensaio da seção 4, com astigmatismo de 1,7–2,5 D, fecha a questão.

## 4. Segundo lote (18/10/2026): três pacientes, astigmatismo até 2,5 D

Arquivos recebidos por e-mail (zip), exportados da mesma estação: pasta com 6 exames (os 4 do paciente A e
2 de um paciente B, 05/08/2026), pasta com os mesmos 4 exames do paciente A, e uma pasta com quatro subpastas
`topo 1..4`, cada uma com os 4 exames de um paciente C (28/11/2025). Nenhum nome entrou no projeto.

**Orientação angular — ensaio decisivo.** Meridiano mais curvo calculado (eixo do cilindro negativo − 90°)
contra SimK1 Ang. e eixo Fourier 6 mm do próprio TMS:

| Exame | Olho | CYL TMS (D) | SimK íngreme | Fourier 6 mm | Calculado 4 mm | Calculado 6 mm | Δ |
|---|---|---|---|---|---|---|---|
| 56301 | OD | 2,48 | 82° | 82° | **83°** | 83° | 1° |
| 56302 | OS | 2,09 | 111° | 111° | **110°** | 111° | ≤ 1° |
| 27201 | OD | 1,69 | 87° | 90° | **88°** | 89° | ≤ 2° |
| 27203 | OD | 1,65 | 89° | 91° | **89°** | 89° | ≤ 2° |
| 27202 | OS | 0,80 | 84° | 85° | 88° | 88° | 3–4° |
| 27204 | OS | 0,75 | 79° | 85° | 88° | 89° | 3–9° (TMS varia 6° entre SimK e Fourier) |

Com astigmatismo ≥ 1,6 D a diferença fica dentro de 2° em 4 de 4 exames; espelhamento (180° − θ) ou rotação de
90° dariam 97°/70°/92°/91° e 173°/20°/178°/179° — descartados. **Convenção angular fechada.**

**Magnitude.** Cilindro refrativo calculado (n = 1,376, zona de 4 mm) / CYL ceratométrico do TMS (SimK, anéis 6–8, ≈ 2,1–2,9 mm):
2,78/2,48 = 1,12; 2,47/2,09 = 1,18; 1,89/1,69 = 1,12; 1,91/1,65 = 1,16 — compatível com o fator 0,376/0,3375 = 1,114
mais a diferença de zona. O "6mm Reg. Astig." do XREF é a *amplitude* de Fourier (≈ metade do cilindro).

**Outros achados.**
- Exame 56301: `PUP_1_56301.dat` com 256 zeros — o TMS não detectou a borda pupilar nessa captura. O programa
  marcou a pupila como *ausente* e deixou os perfis pupilares **indisponíveis** com o motivo registrado; os perfis
  de vértice saíram normalmente. (Primeiro caso real do terceiro estado da §7.)
- Paciente B (56301/56302): 0 pontos interpolados, resíduo 0,22–0,28 µm, R 7,25/7,30 mm, Q −0,31/−0,28; SA 6 mm
  +0,369/+0,341 µm; HOA RMS 0,46/0,41 µm.
- Paciente C, 2 capturas por olho: SA 6 mm OD +0,249/+0,254, OS +0,339/+0,336 µm (Δ ≤ 0,005); trefoil 0,225/0,222
  e 0,075/0,094; coma OD 0,300/0,265, OS 0,200/0,042 — coma volta a ser o termo instável entre capturas.
- **Exportar duas vezes para a mesma pasta acumula pacientes**: o TmsSw acrescenta as linhas ao `XREF.dat`
  existente (um cabeçalho, 6 linhas) e os arquivos `*_1_<exame>.dat` convivem. O programa agora deriva o código de
  estudo do XREF (hash do nome/ID, sem armazenar o nome), conta capturas por paciente e olho, avisa na abertura e
  põe o código no nome dos relatórios. O campo *File REF.* vale 1 em todas as linhas e o *ID* vem vazio — não servem
  para separar pacientes.
- Exportações repetidas do mesmo exame (4 subpastas, 33 arquivos cada) são **byte a byte idênticas** → a identidade
  do cálculo (`analysis_id`) é a mesma, como exige a §4.

## 4b. SimK e ACP do TMS reproduzidos (18/10/2026, versão 0.4.0)

Com a ceratometria por zona (D23), o programa recalcula a cada análise o SimK pela regra do TMS-4N — média dos anéis
6–8 (≈ Ø 2,5 mm) em cada meridiano completo; curvo = máximo; plano = +90° — e a média por área dentro do contorno
pupilar. Nos 10 exames distintos: SimK curvo/plano a ≤ 0,01 D do XREF em 9 de 10 e a 0,03 D no 10º (27204, CYL
0,75 D, eixo do meridiano máximo 4° fora); **Km da zona pupilar × ACP: erro médio 0,006 D (máx 0,017)**. O eixo do
2º harmônico (a 3 mm) repete entre capturas a 0,1–3°, contra 2–13° do eixo do SimK. O SimK do TMS-4N é uma zona de
≈ 2,1–2,9 mm, não de 3 mm. O "3mm Reg. Astig." do XREF é a **amplitude** do 2º harmônico (metade do cilindro): 2 × amplitude
fica a ≤ 0,11 D do ΔK da zona 0–3 mm nos 10 exames, e o eixo a 1–2° do meridiano curvo quando ΔK ≥ 0,75 D.

## 5. Pendências

- Confirmar modelo/versão do software na tela "Sobre" do TmsSw (o formato indica TMS-4/4N com cone de 25).
- Olho de calibração (esfera de raio conhecido), se existir no fluxo da Tomey — teste de unidade mais limpo.
- Comparador (Pentacam/OPD) do mesmo olho para o G3.
- Repetibilidade: as 2 capturas por olho mostram SA estável (Δ ≤ 0,02 µm) e coma variável (Δ até 0,16 µm);
  o piloto (§9) quantifica com 3 capturas e 20–30 participantes.
