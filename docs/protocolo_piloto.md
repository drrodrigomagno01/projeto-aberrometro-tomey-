# Protocolo do piloto exploratório (plano §9) — modelo para preencher

> Rascunho operacional. Aprovações éticas/institucionais, termo de consentimento e base legal (LGPD) devem ser
> definidos antes de qualquer coleta nova. O piloto estima variabilidade e falhas; **não** demonstra
> intercambialidade clínica.

## 1. Pergunta e desfechos

- Pergunta primária: a aberrometria corneana anterior **calculada** a partir do TMS é repetível e qual é a sua
  concordância com o comparador para c(4,0), coma, trefoil e RMS HOA a 4 e 6 mm?
- Desfechos: desvio-padrão intrassujeito (Sw) e coeficiente de repetibilidade por método; viés e limites de
  concordância (Bland–Altman com observações repetidas) por métrica; taxa de importação e de perfis
  disponíveis; motivos de falha; dependência do erro em relação à magnitude; resultados por diâmetro, grupo
  topográfico e centro.
- Unidade de análise: **participante** (um olho por pessoa, sorteado/previamente definido). Se ambos os olhos
  forem usados, modelo com agrupamento (não implementado no `tmsab stats`; filtrar um olho).

## 2. Comparador (definir ANTES da coleta)

| Item | Preencher |
|---|---|
| Equipamento comparador | ☐ Pentacam (frente de onda corneana anterior) ☐ OPD-Scan (aberrações corneanas) ☐ outro |
| Versão do software e relatório usado | |
| Grandeza reportada: superfície anterior apenas? total? | |
| Abertura e plano (4/6 mm, corneano/pupilar) | |
| Centro (vértice, pupila, outro) | |
| Índice de refração e referência de onda | |
| Normalização e convenção de sinal (OSA?) | |
| Termos incluídos no RMS HOA | |
| Diferenças de definição que impedem comparação direta | |

Se alguma definição ficar desconhecida, a análise é "comparação entre métodos" com essa limitação declarada.
**Não** ajustar parâmetros do programa retrospectivamente para fazer os números coincidirem.

## 3. Participantes e aquisição

- 20–30 participantes; grupos descritos separadamente: topografia regular; astigmatismo ≥ 1,5 D; irregular
  (ceratocone/pós-cirúrgico) — este último só para explorar limites, sem ampliar a alegação de uso.
- Três capturas por aparelho por sessão, com reposicionamento entre capturas; comparador em condições
  padronizadas e temporalmente próximas; variar a ordem dos aparelhos; segunda sessão ou segundo operador em
  subconjunto predefinido, se possível.
- Registrar: operador, sequência, iluminação, condição pupilar, qualidade de alinhamento, observações do
  filme lacrimal, índices de qualidade do TMS.
- Exportar cada captura em pasta própria: `E###_<OD|OS>_c<n>/` com manifesto (`tmsab manifest … --capture n`).

## 4. Processamento

```
tmsab batch COLETA/ --out SAIDA/            # relatórios por captura + resumo_longo.csv
# anexar as linhas do comparador ao CSV longo (metodo = "Pentacam_ant" ou "OPD_corneal"), mesmas métricas
tmsab stats SAIDA/resumo_longo.csv --out ESTATISTICA/ --compare TMS_calc,Pentacam_ant --metrics sa_c40@6mm_vertice,coma_magnitude@6mm_vertice,rms_hoa_n3plus@6mm_vertice
```

Desfecho principal deve corresponder ao fluxo de uso pretendido (captura única × média de três — declarar).
Congelar versão do programa (`analysis_id` registra versões) antes de analisar o conjunto reservado.

## 5. Critérios predefinidos (preencher com o responsável clínico e o estatístico)

| Métrica | Diferença clinicamente relevante (margem) | Precisão desejada dos LoA | Observações |
|---|---|---|---|
| c(4,0) @ 6 mm | | | |
| coma @ 6 mm | | | |
| RMS HOA @ 6 mm | | | |
| c(4,0) @ 4 mm | | | |

## 6. Saídas do piloto

- Relatório exploratório com Sw, CR, viés, LoA (e ICs), gráficos de Bland–Altman, taxas de falha e motivos.
- Limites por configuração (diâmetro × centro × grupo) que o produto deverá restringir explicitamente.
- Dimensionamento do estudo confirmatório (§9 "Estudo confirmatório").
