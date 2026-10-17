# Verificação científica e de software (plano §8)

Quatro níveis: (1) interpretação dos arquivos; (2) fidelidade geométrica; (3) cálculo óptico e numérico;
(4) desempenho em aquisições reais. Os níveis 1–3 estão cobertos por `pytest` (53 ensaios) e por `tmsab verify`.
O nível 4 depende de dados reais (G0) e do piloto (§9). Aprovar um nível não aprova o seguinte.

## Matriz mínima de ensaios → implementação

| Ensaio (plano) | Teste | Critério registrado | Resultado atual |
|---|---|---|---|
| Leitura lexical | `test_adapter.py`: truncado, célula corrompida, unidade errada, decimal vírgula, arquivo extra | erro tipado (`AdapterError`) com causa e requisito; nenhuma carga parcial | passa |
| Angle e Ring | `test_roundtrip_angle_and_ring_orientations_equivalent` | exames normalizados idênticos após transformação | passa |
| Pareamento | mistura de olhos, conflito manifesto×nome, tabela obrigatória ausente | rejeição antes do cálculo | passa |
| Sinais e ausências | `test_states_and_sign_rules` | zero-ausente vira NaN/estado; negativo-interpolado vira valor + estado; nunca valor absoluto global | passa |
| Orientação | `test_mirror_flips_horizontal_coma_sign_only`, `test_rotation_by_90_moves_coma_between_components`, tórica a 0° e 45° | espelho em x inverte c(3,+1) e c(2,−2), mantém c(3,−1) e c(4,0); rotação 90° leva c(3,+1)→c(3,−1); eixo do astigmatismo recuperado | passa (< 10⁻⁶ µm) |
| Geometria | `test_conic_fit_recovers_parameters`, `test_reconstruction_residual_and_holdout` | R, Q exatos para cônica; resíduo RMS < 2 nm com exportação de 6 casas; pontos reservados (anéis 5 e 15) com resíduo < 20 nm; cônica+Zernike × cônica+RBF concordam < 0,1 µm em 4 mm | passa |
| Base de Zernike | `test_orthonormality_on_disk`, `test_gradient_matches_numeric`, `test_fit_recovers_known_coefficients_with_noise` | ortonormalidade < 5×10⁻³ (discretização); gradiente analítico = numérico < 10⁻⁶; recuperação < 3×10⁻³ µm com ruído 0,01 µm | passa |
| Traçado de raios | `test_descartes_oval_has_zero_wavefront`, `test_sphere_has_positive_primary_sa_matching_seidel`, `test_conic_q_dependence_matches_seidel_factor`, `test_3d_trace_matches_independent_meridional_reference` | oval de Descartes Q=−1/n²: |W| < 10⁻⁴ µm (obtido 3×10⁻⁶); esfera 7,8 mm a 3 mm: c(4,0)=+0,0204 µm vs Seidel +0,0197 (3,5%, diferença = SA de ordem superior); razão de SA entre Q=−0,26 e Q=0 = (1/n²−0,26)/(1/n²) ± 2%; traçado vetorial 3D × traçado meridional escalar independente < 10⁻⁴ µm (obtido < 10⁻⁵) | passa |
| Convergência | `test_convergence_with_ray_grid` (49/97/193 raios por lado) | Δc(4,0), Δc(3,1) < 2×10⁻⁴ µm entre 97 e 193 | passa |
| Lacunas e deslocamento | `test_missing_sector_blocks_6mm_but_not_4mm`, `test_interpolated_sector_is_used_but_reported`, `test_pupil_centered_profile_shows_decentration_coma` | setor ausente ≥ 60° bloqueia 6 mm e preserva 4 mm; interpolado é usado mas reportado; deslocamento pupilar (+0,3, −0,2) mm gera coma com sinal do deslocamento e magnitude da verdade analítica (< 3×10⁻³ µm) | passa |
| Saídas | `test_exports_consistent`, `test_blocked_profile_reported_as_absence` | JSON, CSV e objeto com o mesmo c(3,1); perfil bloqueado reportado como ausência (não zero); JSON sem NaN | passa |
| Distribuição | workflow `build-windows.yml` + `build_windows.bat` | executável roda `verify`, `synth`, `analyze` sem rede | a executar no runner Windows |

## Casos ópticos que evitam falsos sucessos

- Esfera **não** dá resultado nulo: dá SA positiva prevista — exigido pelo teste.
- Simetria axial (esfera/cônica, objeto axial, abertura coaxial): todos os termos não simétricos < 10⁻⁹ µm.
- Oval de Descartes: único caso com W ≡ 0 — usado como teste de nulidade exata do traçado e da referência.
- Testes de polinômios (base) e de superfícies (traçado) são separados.
- Unidade deliberadamente errada (RAD ×1000) é bloqueada pela faixa física, não convertida.

## Orçamento de erro — primeiros números (sintéticos, superfície Q=−0,2, R=7,8 mm, 6 mm, c(4,0) verdadeiro = 0,2204 µm)

| Fonte | Condição | Efeito em c(4,0) a 6 mm |
|---|---|---|
| Quantização da exportação | 6 casas (1 nm) | +0,0001 µm |
| | 4 casas (0,1 µm), sem ruído | **+0,004 µm** (erro sistemático: anéis quantizados de forma coerente) |
| | 4 casas + ruído 0,3 µm | −0,001 a −0,002 µm (o ruído "dithera" a quantização) |
| | 3 casas (1 µm), sem ruído | **−0,035 µm (−16%)** — inaceitável para decisões a 0,1 µm |
| | 3 casas + ruído 0,3 µm | −0,012 µm |
| | 2 casas (10 µm) | perfil **bloqueado** (resíduo 2,9 µm > 1 µm) |
| Ruído de altura | 0,3 µm RMS por ponto | ±0,002 µm (média sobre ~3,3 mil pontos) |
| | 0,5 µm RMS + 3 casas | −0,014 µm |
| Amostragem de raios | 97 vs 193 por lado | < 2×10⁻⁴ µm |
| Ordem da superfície | 10 (padrão) | resíduo limitado pela quantização |

Conclusão operacional (vai ao G0): **exigir ≥ 4 casas decimais** na exportação de HIT/RAD; registrar a precisão
no manifesto; o relatório mostra o resíduo da reconstrução, que denuncia exportações grosseiras.

Faltam (dependem de dados reais): erro da reconstrução arc-step do próprio TMS, estabilidade das derivadas em
córneas irregulares reais, registro pupilar, variabilidade de aquisição (filme lacrimal, alinhamento) — todos
entram pelo piloto (§9) e pelo estudo de repetibilidade.

## Tolerâncias numéricas congeladas na versão 0.1.0

- Oval de Descartes: |W| < 10⁻⁴ µm. Traçado 3D × meridional: < 10⁻⁴ µm. Convergência de grade: < 2×10⁻⁴ µm.
- Recuperação sintética (6 casas): |Δc| < 2×10⁻³ µm para SA, coma, trefoil; < 10⁻² µm para astigmatismo.
- Reconstrução (6 casas): resíduo RMS < 2 nm; pontos reservados < 20 nm.

Esses limites são de **verificação numérica**; permanecem separados dos limites clínicos de concordância, que
só o estudo do §9 pode estabelecer.
