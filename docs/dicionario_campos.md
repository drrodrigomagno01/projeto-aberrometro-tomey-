# Dicionário de campos — exportação "Data Tables" do TMS-4/4N

> **Formato real verificado em 18/10/2026** (perfil `tms4_datatables_v1`): ver a tabela em `docs/G0_resultado.md`. O texto abaixo é o dicionário presumido original (perfil v0), mantido para os pacotes sintéticos e para registro das hipóteses.


Fonte documental: manual TMS-4N §3.11.5 e manual TMS-4 §3.11.5 (referências [1] e [2] do plano); guia Precilens [3].
Tudo abaixo é **hipótese de trabalho** até a verificação com arquivos reais. A coluna "Verificado" deve ser
preenchida no G0/G1.

| Campo | Descrição documentada | Unidade presumida | Uso no programa | Regra de decodificação presumida | Verificado |
|---|---|---|---|---|---|
| HIT | Profundidade sagital referida ao ápice (plano apical) | mm, + posterior | Altura z da superfície (reconstrução) | 0 → ausente; negativo → interpolado (valor = módulo) | ☐ |
| RAD | Raios de vídeo corrigidos = posição radial física do ponto | mm (no plano x-y) | Coordenada r do ponto | 0 → ausente; negativo → interpolado | ☐ |
| ROC | Raio de curvatura instantâneo (meridional) | mm | Verificação geométrica auxiliar (não entra no cálculo) | idem | ☐ |
| DIO | Potência dióptrica (convenção ceratométrica, 337,5/ROC) | D | Comparação topográfica, só conferência | idem | ☐ |
| PUP | Pupila digitalizada: centro e contorno referidos ao ápice | mm | Centro de análise do perfil "pupila" | arquivo chave=valor: `center_x_mm`, `center_y_mm`, `diameter_mm`, `frame`, `method` | ☐ |
| RNG | Distância no plano de vídeo | pixels | Proveniência apenas | não usado | ☐ |
| XREFDAT | Índice pessoa↔exame | — | **Não importado** (dados nominais ficam fora do ambiente de análise) | ignorado | — |

## Estrutura das tabelas (presumida)

- 256 posições angulares por anel; 25 anéis (TMS-4) ou 31 (TMS-4N com cone de 31).
- Orientação **Angle**: 256 linhas × N colunas (uma linha por ângulo). Orientação **Ring**: N linhas × 256 colunas.
  O leitor detecta pela forma; tabelas 256×256 são ambíguas e exigem `orientation` no perfil.
- Primeira posição angular às **3 horas** (0°), avançando no sentido **anti-horário** na vista do observador — a confirmar.
- Coluna de índice (0..255 ou 1..256) opcional; linhas de cabeçalho (texto) são puladas antes do início dos dados.
- Ausente/interpolado: o manual descreve marcação "por sinal ou zero"; a regra é **por campo** (nunca valor absoluto global).

## Nome dos arquivos (presumido)

`<EXAME>_<OD|OS>_<HIT|RAD|ROC|DIO|PUP>.<csv|txt>`; o olho pode vir do manifesto. Conflito entre nome e
manifesto bloqueia a importação (ensaio *Pareamento*).

## Manifesto (`manifest.json`, criado por `tmsab manifest`)

Campos: `study_id` (código pseudonimizado), `exam_id`, `eye`, `acquired_at`, `date_precision`, `capture_index`,
`device{model, software_version, cone, calibration_state}`, `export{orientation, decimals, dialog_screenshot}`,
`synthetic` (false para exames reais). **Nunca** incluir nome, data de nascimento ou outros identificadores diretos.
