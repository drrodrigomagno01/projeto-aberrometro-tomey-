# Convenções físicas e ópticas (plano §4–§5)

## Sistema canônico de coordenadas

- Vista do **observador** (examinador olhando o olho): **x para a direita da imagem, y para cima**, ângulo θ
  **anti-horário** a partir das 3 horas (θ = 0°). **z positivo para dentro do olho** (posterior).
- Origem no **vértice topográfico** (ápice do mapa do TMS). O centro pupilar e o centro de análise são campos
  distintos; a ausência de pupila **não** é preenchida com o vértice.
- OD e OS mantêm a lateralidade original; nenhuma simetrização ou espelhamento automático. Em OD, +x é
  nasal; em OS, +x é temporal (na vista do observador).
- Reflexão e rotação são operações distintas (`MirroredSurface`, `RotatedSurface`), usadas em ensaios de sinal.

## Grandezas

- **Altura** (superfície): z = S(x, y) em **mm**, modelo "cônica + resíduo de Zernike (ordem 10)" ou "cônica + RBF".
- **Frente de onda** (óptica): W(x, y) em **µm**, definida sobre uma abertura circular no plano corneano.
  Altura e frente de onda são objetos diferentes; ajustar polinômios à altura não produz aberrações ópticas.
- **Asfericidade Q** e raio R vêm do ajuste cônico (linear: r² = 2Rz − (1+Q)z²) e são descritores de forma;
  permanecem separados da aberração esférica.

## Modelo óptico "anterior_v1"

| Parâmetro | Valor | Observação |
|---|---|---|
| Sistema | superfície anterior única, ar → córnea | identificado como modelo anterior |
| Índices | n_in = 1,000; n_out = 1,376 | assunções do modelo, não medidas do indivíduo |
| Objeto | axial, no infinito (raios paralelos ao eixo z) | outros conjugados exigem outro perfil |
| Abertura | círculo de diâmetro 4 ou 6 mm **no plano corneano** (z = 0), centro no vértice ou no centro pupilar | declarar plano, centro e diâmetro |
| Raio principal | raio pelo centro de análise; direção refratada t_c | — |
| Potência equivalente | P = (n_out − n_in)·κ_média, κ do Hessiano da superfície no centro (média das curvaturas principais) | aproximação paraxial local |
| Ponto imagem de referência | F = P_c + s'·t_c, com s' = n_out/P | remove tilt por construção |
| Esfera de referência | centro F, raio s', passa por P_c | registrada em `reference` |
| Caminho óptico | OPL = n_in·z(x,y) + n_out·s, com s a distância assinada de P ao ponto da esfera **do lado próximo de F** ao longo do raio refratado (extensão virtual permitida — definição clássica em espaço imagem) | — |
| Sinal | **W = OPL_principal − OPL(raio)**; positivo = frente de onda adiantada | esfera refratora ⇒ c(4,0) > 0; oval de Descartes (Q = −1/n²) ⇒ W ≡ 0 |
| Refoco opcional | `refoco_rms`: s' que minimiza o RMS de W após remover pistão e tilt; registrado nas notas | não é o padrão |
| Dispersão | nenhuma | sem alegação espectral |
| Amostragem | grade cartesiana 97×97 no quadrado circunscrito, filtrada ao disco (≈7,2 mil raios) | convergência verificada (49/97/193) |

## Zernike

- Convenção **OSA/ANSI Z80.28** (Thibos 2002): índices (n, m), j = (n(n+2)+m)/2, base **ortonormal** no disco
  unitário, ρ = r/a, θ anti-horário.
- Ordem radial máxima **6** (28 termos) por padrão; ajuste por mínimos quadrados (SVD) com condicionamento reportado.
- RMS HOA = raiz da soma dos quadrados dos termos com n ≥ 3 (válido como RMS no disco apenas com base
  ortonormal e abertura completa — o suporte é verificado antes).
- Termos **removidos** do RMS HOA: pistão, tilt, defocus, astigmatismo (todos informados na tabela).
- Componentes assinados preservados: c(4,0); c(3,−1), c(3,+1); c(3,−3), c(3,+3). Magnitudes agregadas =
  raiz da soma dos quadrados dos pares.
- Mudança de abertura exige novo traçado e novo ajuste (nunca reescalonamento de coeficientes).
- Cilindro/eixo "equivalentes" derivam de c(2,±2) pelos vetores de potência (J0 = −2√6·c22/a², J45 = −2√6·c2−2/a²),
  em dioptrias de potência refrativa da frente de onda (n = 1,376) — **não** são a ceratometria (337,5/R).

## Ceratometria por zona (D23, `tmsab/keratometry.py`)

- **Fonte**: DIO das Data Tables = potência **axial** (sagital) do TMS, convenção ceratométrica 337,5/R_axial com
  R_axial = r·√(1+s²)/s (s = inclinação radial). Verificado nos exames reais (DIO·R_axial/337,5 = 1,000 ± 0,0005). Não é
  a curvatura local naquele diâmetro (essa é a tangencial, ROC) nem a potência refrativa da frente de onda (n 1,376).
- **Ajuste**: P(θ) = a0 + a1·cosθ + b1·senθ + a2·cos2θ + b2·sen2θ por mínimos quadrados. K médio = a0; h2 = √(a2²+b2²);
  **K2 (curvo) = a0 + h2; K1 (plano) = a0 − h2**; ΔK = 2·h2; eixo do K2 = ½·atan2(b2, a2); eixo do K1 = eixo do K2 + 90°.
  Assimetria = amplitude do 1º harmônico; irregularidade = RMS do resíduo após os harmônicos 0–2.
- **Eixos** em graus TABO (0° às 3 h, anti-horário na vista do examinador, nos dois olhos — o referencial canônico),
  registrados em (0, 180]: 0° vira 180°.
- **Circunferência** de diâmetro d, centro no vértice: em cada um dos 256 meridianos, DIO interpolado linearmente em
  r = d/2 entre os anéis vizinhos com o RAD daquele ângulo (os anéis se deslocam com a curvatura e com o ângulo).
- **Diâmetros padrão** (`KeratometryConfig`): Ø 1, 2, 2,4, 3, 3,3, 4, 5, 6, 7 e 8 mm. 2,4 mm ≈ anel dos biômetros
  ópticos (IOLMaster 2,3–2,5 mm, Lenstar 2,3 mm, AL-Scan 2,4 mm) e do SimK; 3,3 mm = anel externo do AL-Scan, que previu
  melhor o astigmatismo residual de LIO tórica que o de 2,4 mm (Chang 2022). A folha A4 mostra Ø 1–8, a zona 0–3 mm, a pupila e o SimK;
  as tabelas completas (programa, HTML, relatório, CSV) levam todas as linhas e os 25 anéis.
- **Anel medido**: valores do próprio anel; diâmetro = 2·média do RAD medido (= 2ª coluna do AVE).
- **Zona 0–d** (padrão 0–3 e 0–6 mm): área de raio d/2 centrada no vértice, com a mesma regra da zona pupilar (peso
  de área da célula radial, Km = média ponderada); equivale às "zonas" de Pentacam/Atlas. Com a pupila centrada no vértice, zona 0–d =
  zona pupilar de diâmetro d (ensaio). O peso r é maior nos meridianos planos (anéis mais longe do vértice), então o Km
  por área fica alguns centésimos abaixo da média dos anéis quando há astigmatismo. Um meridiano conta como coberto
  quando os anéis com dados chegam à borda (meio anel de folga) e não há ponto ausente dentro dela; como o RAD do ponto
  ausente vem vazio, a posição dele é estimada pelo anel anterior do mesmo meridiano mais o espaçamento nominal.
- **Zona pupilar**: área dentro da borda do PUP no mesmo meridiano. Cada amostra representa a célula radial
  [r − Δr/2, r + Δr/2] (Δr = espaçamento nominal do anel) e pesa a área dessa célula dentro da borda: r·Δr quando
  inteira dentro, só a parte de dentro na amostra da borda (ceratometria_v2, 0.4.1 — antes a amostra entrava inteira
  se r ≤ borda, e a média dava degraus de até 0,15 D de Ø 2 mm em diante quando um anel entrava pelos meridianos
  curvos; D27). Km = média
  ponderada (definição do ACP do TMS: erro médio 0,0045 D nos 9 exames reais com pupila); K1/K2 = Km ∓ h2 do ajuste
  ponderado. Pupila de entrada sob a
  iluminação do Plácido, na captura — não é pupilometria mesópica/escotópica.
- **SimK do TMS-4N** (reproduzido): média dos anéis 6–8 (≈ Ø 2,1–2,9 mm, centro em ~2,5 mm) em cada meridiano completo
  (θ e θ+180°); curvo = máximo; plano = meridiano a +90°; MinK = mínimo em qualquer ângulo. **No XREF, SimK1 = curvo.**
- **Alerta "astigmatismo dependente da zona"** (informativo, em `alertas` e nas notas): diferença vetorial entre o
  astigmatismo da zona 0–3 mm e o da zona pupilar > `k_zona_alerta_dif_D` (0,50 D), ou eixos a mais de
  `k_zona_alerta_eixo_deg` (10°) com ΔK ≥ `k_eixo_min_cyl_D` (0,75 D) nas duas. Base: a zona de 3 mm no vértice foi a
  que melhor previu o astigmatismo em LIO tórica (Meng 2021); com pupila grande a divergência importa. Não escolhe K.
- **J0/J45** (Thibos, positivo = a favor da regra): J0 = −(ΔK/2)·cos(2·eixo K2) = −a2; J45 = −(ΔK/2)·sen(2·eixo K2) = −b2.
  Estatística (resumo_longo.csv) usa J0/J45 e K médio, nunca o eixo cru.
- **Qualidade** (limites provisórios em `QualityConfig`): meridianos (ou meridianos da pupila) com dados ≥ 85% e maior
  setor sem dados ≤ 20°, senão a linha fica indisponível com motivo; marca † com ≥ 10% dos meridianos usando pontos
  interpolados pelo aparelho; nota "eixo pouco definido" com ΔK < 0,50 D; conferência SimK recalculado × XREF com
  tolerância de 0,05 D e 5° (eixo só com CYL ≥ `k_eixo_min_cyl_D`, 0,75 D).

## Chord mu aparente (`tmsab/pupila.py`)

- **Definição** (Chang & Waring 2014): distância entre o centro da pupila e o reflexo corneano coaxial com o olho
  fixando o alvo. No TMS, o vértice (origem do mapa) é esse reflexo: o centro dos anéis com o olho no alvo de
  fixação; o centro da pupila vem do ajuste de círculo à borda do PUP. **Chord mu aparente = |centro da pupila −
  vértice|**, com as componentes no referencial do exame (x, y) e anatômicas: **nasal** (+ nasal, − temporal; em OD
  +x é nasal, em OS +x é temporal) e **superior** (+ superior). Olho não identificado: só x e y.
- **Pupila desta captura** (iluminação do Plácido, 5,8–7,8 mm nos exames reais): o centro da pupila se desloca com o
  tamanho dela; a medida pressupõe boa fixação no alvo. Sem pupila registrada não há chord mu (nunca usa o vértice).
- Entra no objeto da análise (`chord_mu`, JSON), na folha A4 (bloco PUPILA), na aba Ceratometria e na Importação do
  programa, no relatório em texto, nas linhas de métrica `pupila_desta_captura` do CSV, nos detalhes do relatório HTML (linha da
  pupila) e no resumo devolvido pela página (`summary["chord_mu"]`), e no `resumo_longo.csv`
  (`chord_mu_aparente_mm`, `chord_mu_nasal_mm`, `chord_mu_superior_mm`: componentes anatômicas, comparáveis entre olhos).
  Nos 9 exames reais com pupila detectada: 0,11–0,24 mm; entre as duas capturas do mesmo olho a distância difere
  ≤ 0,02 mm e o vetor ≤ 0,03 mm.

## Saídas da ceratometria e do chord mu (0.4.0)

- **JSON** (`Analysis`): `keratometry` (config, `zonas` = circunferências, `discos` = zonas 0–d, `aneis`, `pupila`,
  `simk_calculado`, `simk_tms` só com as colunas numéricas permitidas do XREF, `conferencia`, `alertas`, `notes`) e
  `chord_mu` (`distancia_mm`, `x_mm`, `y_mm`, `nasal_mm`, `superior_mm`, `diametro_pupila_mm`; null sem pupila).
- **CSV por exame** (`<stem>.csv`, 18 colunas, retangular como na 0.3.1): coeficientes e métricas por perfil; no fim,
  linhas de métrica com perfil `pupila_desta_captura` (`pupila_estado`, `diametro_pupila_mm`, `chord_mu_aparente_mm`,
  `chord_mu_x_mm`, `chord_mu_y_mm`, `chord_mu_nasal_mm`, `chord_mu_superior_mm`).
- **CSV da ceratometria** (`<stem>_ceratometria.csv`, 21 colunas, retangular): identificação, tipo, zona, diametro_mm,
  validade, k1_plano_D, eixo_k1_deg, k2_curvo_D, eixo_k2_deg, delta_k_D, k_medio_D, j0_D, j45_D, assimetria_D,
  irregularidade_D, fracao_cobertura, fracao_interpolados, motivo — uma linha por circunferência, zona 0–d, pupila,
  SimK recalculado, SimK do XREF e anel. Gravado ao lado dos outros relatórios e baixado na versão HTML.
- **resumo_longo.csv** (`tmsab batch`): `k1_plano_D`, `k2_curvo_D`, `delta_k_D`, `k_medio_D`, `j0_D`, `j45_D` com o sufixo
  `@{d}mm_vertice`, `@zona_0-{d}mm_vertice` ou `@pupila` (`validade@…` quando indisponível), `diametro_pupila_mm`,
  `chord_mu_aparente_mm`, `chord_mu_nasal_mm`, `chord_mu_superior_mm`. Nunca o eixo cru da ceratometria.

## Controle de diâmetro da ceratometria (0.4.1)

- **O que mostra:** para o diâmetro escolhido d, a **circunferência Ø d** (DIO interpolado em r = d/2 em cada
  meridiano, com o RAD daquele ângulo) e a **zona 0–d** (média pela área), centradas no vértice — as mesmas
  definições, funções e limites das linhas "Ø d mm" e "zona 0–d mm" da tabela (`ceratometria_no_diametro`), com a
  configuração e os limites de qualidade da própria análise. Nos diâmetros da tabela os números são idênticos. As
  duas variam de forma contínua com d (a zona pelo recorte da célula radial na borda, D27). Um meridiano só conta
  como coberto na zona se tiver ao menos uma amostra com parte da célula dentro da borda.
  Abaixo de ~1 mm a zona tem só 1–2 anéis: K1/K2/ΔK mudam depressa quando o 2º anel entra (até 0,1 D em ΔK
  entre décimos vizinhos nos exames reais) e o eixo ali é pouco confiável. Km/K1/K2 das zonas têm viés de
  ≈ +ΔK²/(8·Km) em relação à média por área exata (Δr médio do anel; ≤ 0,015 D nos exames reais, D27).
- **Faixa e passo:** de 0,5 mm até o Ø nominal do anel mais externo arredondado para cima (8,7–9,0 mm nos exames
  reais), em décimos exatos de mm; o diâmetro inicial é 3,0 mm. Na lacuna central do Plácido, fora da cobertura dos
  anéis ou com setor sem dados, a linha sai indisponível com o motivo (nunca extrapolada, nunca zero).
- **Desenho:** sobre o mapa de potência axial (DIO), o círculo Ø d (cinza tracejado quando a circunferência está
  indisponível) e os meridianos **K2 (curvo, laranja)** e **K1 (plano, azul)** nos eixos TABO da circunferência — no
  referencial do mapa (vista do examinador, x à direita, y para cima) o meridiano a° passa pelo vértice na direção
  (cos a, sen a) —, tracejados quando o eixo é pouco definido (ΔK < 0,50 D); "+" no vértice; o contorno tracejado fino
  é a pupila.
- **Interação:** no programa, controle deslizante (setas: 0,1 mm; PgUp/PgDn: 0,5 mm), caixa numérica ou clique/arraste
  no mapa (ignorado com o zoom ou o arrasto da barra de ferramentas ativos); a escolha vale para os exames seguintes
  (limitada à faixa de cada um) e aparece como linha traço-ponto nos "Gráficos por diâmetro". No HTML, o mesmo com o
  controle deslizante ou o ponteiro (mouse ou toque) sobre o mapa: a grade de diâmetros vem calculada do Python (bloco
  JSON) e o JavaScript só exibe; o relatório .html salvo reabre no diâmetro mostrado e o controle continua funcionando
  nele; a impressão leva o mapa com o diâmetro atual e as duas linhas.
- **O que não muda:** o controle é só visualização — não entra no `analysis_id` nem no JSON, no CSV e no PDF; a tabela,
  a folha A4 e os relatórios continuam com os diâmetros fixos (D23, D26).

## Identidade do cálculo

`analysis_id` = SHA-256 (24 hex) de {hashes dos arquivos de origem, configuração de reconstrução, limites de
qualidade, perfis ópticos, configuração da ceratometria, versões do núcleo/adaptador/NumPy/SciPy, referencial}. Data de execução é campo
separado. Reprocessar com outro centro, diâmetro, método ou versão gera nova análise.
