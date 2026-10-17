# G0 — Fonte confirmada (checklist para fechar o marco)

Critério do plano: *"fonte numérica e campos essenciais disponíveis"*. Enquanto este checklist não estiver
completo, o programa opera em **modo presumido** (aviso em todo relatório) e a compatibilidade **não é declarada**.

## 1. Inventário da estação (tarefa T01)

- [ ] Modelo do aparelho: TMS-4 / TMS-4N / TMS-5 (foto da etiqueta ou tela "Sobre")
- [ ] Versão do software TMS instalado (tela "Sobre"/"About")
- [ ] Número de anéis do cone usado (25 ou 31) e pontos por anel (256)
- [ ] Sistema operacional do PC de aquisição e se há porta USB/rede para retirar arquivos
- [ ] Recursos licenciados visíveis no menu (Fourier, Height, Contact Lens etc.)

> Se o aparelho for **TMS-5**: perguntar à Tomey se o software já entrega análise de Zernike/frente de onda
> corneana antes de prosseguir (pode encerrar o projeto antes de começar). O TMS-4N oficialmente só traz
> análise de Fourier.

## 2. Pacote inicial de exportação (tarefa T02)

Exportar **o mesmo exame** nas variantes abaixo, cada variante em uma pasta própria, sem nomes de pacientes
nos nomes de arquivo (use o código do estudo, ex.: `E001_OD_HIT.csv`):

- [ ] Tabelas **HIT** (altura), **RAD** (posição radial), **ROC** (raio de curvatura), **DIO** (potência) — e **PUP** se existir
- [ ] Orientação **Angle** (linhas por ângulo) e **Ring** (linhas por anel) do mesmo exame
- [ ] Duas precisões, se o software permitir — **mínimo obrigatório: 4 casas decimais em mm** (ver achado abaixo)
- [ ] Um exame OD e um OS
- [ ] Três capturas repetidas do mesmo olho (reposicionando entre capturas)
- [ ] Se existir no fluxo da Tomey: exame do **olho de calibração** (esfera de raio nominal conhecido)
- [ ] Captura de tela (print) da janela de configuração da exportação (separador, decimais, tabelas marcadas)
- [ ] Impressão/PDF do mapa colorido do mesmo exame (para conferir orientação e identificação — não como dado quantitativo)
- [ ] Se possível: Pentacam ou OPD-Scan do mesmo olho no mesmo dia (comparador)

**Achado da verificação sintética (docs/validacao.md):** exportar HIT com 3 casas decimais (1 µm) gera viés
sistemático de até −0,035 µm (−16%) na aberração esférica a 6 mm; com 4 casas (0,1 µm) o erro cai para
≈0,004 µm. Configure a exportação com o máximo de casas disponível.

## 3. Perguntas que o pacote responde (tarefa T03 — dicionário de campos)

Para cada item, anotar a resposta em `docs/dicionario_campos.md` e ajustar `tmsab/adapters/profiles/tms4_datatables_v0.json`:

| Pergunta | Campo do perfil |
|---|---|
| Separador (vírgula, ponto-e-vírgula, tabulação) e separador decimal | `delimiters`, `decimal` |
| Há linha de cabeçalho? Há coluna de índice (ângulo/anel)? | detecção automática — confirmar |
| Primeira posição angular é às 3 horas? Sentido anti-horário na vista do observador? | `angle_start_deg`, `angle_direction` |
| A imagem é vista do observador (x direita = nasal em OD)? Há espelhamento? | `mirror_x` |
| HIT é profundidade sagital a partir do plano apical, positivo posterior? Ou elevação relativa a esfera? | `height.reference`, `height.sign` |
| RAD é distância radial no plano (mm) ou comprimento de arco? | `rad_meaning` |
| Como são marcados dados ausentes (0? vazio? *?) e interpolados (sinal negativo?) | `roles.*.zero_is_missing`, `negative_means_interpolated`, `missing_tokens` |
| Formato do PUP (centro x/y em mm? em pixels? referencial?) | `roles.PUP` + `read_pupil_file` |
| Nome dos arquivos gerados pelo TMS (padrão de HIT/RAD/OD/OS) | `roles.*.patterns`, `eye_patterns`, `exam_id_regex` |

## 4. Critério de saída do G0

- [ ] `tmsab import PASTA` conclui sem erro, com anéis/ângulos corretos, raio médio por anel crescente, HIT na faixa física
- [ ] Mapa de altura na aba *Geometria* coerente com o mapa impresso (orientação: nasal/temporal e superior/inferior corretos)
- [ ] Angle e Ring do mesmo exame produzem o mesmo exame normalizado (ensaio *Angle e Ring*)
- [ ] Um padrão assimétrico conhecido (ex.: olho com astigmatismo oblíquo documentado no mapa impresso) reproduz o eixo esperado
- [ ] Perfil marcado `"status": "verificado"` no JSON e registro da decisão em `docs/decisoes.md`

Depois disso: **G1** (importação aceita) → habilitar reconstrução apenas para o perfil documentado.
