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

## Identidade do cálculo

`analysis_id` = SHA-256 (24 hex) de {hashes dos arquivos de origem, configuração de reconstrução, limites de
qualidade, perfis ópticos, versões do núcleo/adaptador/NumPy/SciPy, referencial}. Data de execução é campo
separado. Reprocessar com outro centro, diâmetro, método ou versão gera nova análise.
