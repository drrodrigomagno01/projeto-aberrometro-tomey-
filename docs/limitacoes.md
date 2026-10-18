# Limitações conhecidas e finalidade (plano §1, §13, §14)

## O que o programa é

Ferramenta de **pesquisa** que calcula a contribuição óptica modelada da **superfície anterior** da córnea a
partir de dados numéricos já adquiridos pelo TMS. Saída: mapa de frente de onda, coeficientes de Zernike,
aberração esférica primária, coma, trefoil e RMS de alta ordem, sempre vinculados a abertura, centro, modelo e
versões.

## O que o programa NÃO é

- Não é aberrometria **ocular total**; não separa aberrações internas (cristalino/LIO) nem usa a superfície posterior.
- Não faz aquisição, não controla o instrumento, não altera o software do TMS.
- Não emite alegação diagnóstica, não seleciona LIO, não planeja tratamento, não atribui "normalidade".
- Não declara equivalência com OPD-Scan, Pentacam ou outro sistema: isso exige protocolo, grandeza e população
  definidos e resultados que a sustentem (plano §9 e §14).
- Não possui registro na Anvisa nem enquadramento declarado (RDC 657/2022); uso interno de pesquisa.
  Qualquer finalidade clínica ou distribuição a terceiros reabre requisitos, risco e validação.

## Limitações técnicas desta versão

1. **Perfil de exportação presumido** (`tms4_datatables_v0`): orientação angular, espelhamento, semântica de
   RAD, referência de HIT e formato de PUP são hipóteses até o G0/G1. O relatório avisa enquanto não houver verificação.
2. A altura exportada é a reconstrução **arc-step do próprio TMS**; erros desse algoritmo propagam e não são corrigíveis aqui.
3. Lacuna central do Plácido (~0,35 mm presumida) e lacunas de pálpebra: regiões sem dados são **estimadas pelo modelo**
   e sinalizadas; perfis com suporte insuficiente ficam **indisponíveis**.
4. Modelo óptico simplificado: superfície única, índice constante 1,376, objeto axial no infinito, abertura no
   plano corneano, raios paralelos ao eixo do videoceratoscópio. Comparações com aparelhos que reportam no plano
   pupilar exigem perfil correspondente e registro.
5. Curvatura paraxial local pelo Hessiano; para centros muito descentrados (> 1,5 mm) a aproximação piora (aviso emitido).
6. Zernike até ordem 6 (28 termos) por padrão; córneas muito irregulares podem deixar informação relevante no
   resíduo (Smolek & Klyce 2003) — o resíduo é mostrado e deve ser inspecionado.
7. Sensibilidade à **precisão da exportação**: 3 casas decimais (1 µm) introduzem viés sistemático (ver docs/validacao.md).
8. Limites de qualidade **provisórios**, derivados do orçamento de erro sintético, não de critérios clínicos.
9. TMS-5 (anterior + posterior) **não** é suportado nesta versão (adaptador próprio e nova validação, plano §15).
10. Estatística do piloto implementa Bland–Altman para observações repetidas com um olho por participante; dois
    olhos exigem modelo com agrupamento (não implementado).

## Dados pessoais

Dados nominais entram **apenas** na página impressa "Resultado completo" (1ª página do PDF, figura do HTML e aba do
programa), por opção do usuário, para o prontuário; não entram no JSON/CSV, nas páginas técnicas nem nos logs — essas
saídas usam códigos de estudo. O código é um
hash do nome/ID presente no `XREF.dat` (calculado na leitura e descartado) — isso é **pseudonimização, não
anonimização**: quem tem a lista de pacientes da clínica pode recompor a correspondência, portanto a pasta
exportada e a subpasta `analises/` continuam sendo dados pessoais sob a LGPD e devem ficar no ambiente da clínica.
O próprio `XREF.dat` exportado pelo TMS contém o nome: não o copie para fora da estação. Dados reais só entram em
desenvolvimento assistido por IA nos ambientes e condições aprovados.

## Versão HTML

Precisa de internet na primeira abertura (motor Pyodide + NumPy/SciPy/Matplotlib do CDN jsdelivr; depois fica em
cache do navegador); os dados dos exames nunca saem do computador. Cálculo ~3× mais lento que o programa nativo.
Safari: a seleção de pasta funciona pelo diálogo de arquivos; o arrastar-e-soltar de pastas depende da versão.
