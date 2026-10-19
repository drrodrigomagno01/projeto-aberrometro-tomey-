#!/usr/bin/env bash
# Publica a versão HTML no Vercel (projeto "aberrometria-tms", time "RODRIGO MAGNO's projects").
#
# Uso (de qualquer pasta):
#   web/publish_vercel.sh            publica o HTML que o CI (GitHub Actions) gerou e testou para o commit atual
#   web/publish_vercel.sh ARQ.html   publica um HTML já gerado (ex.: python web/build_html.py --out dist_html)
#
# Requer: gh logado no GitHub (modo padrão), Node/npx e o Vercel CLI logado (npx vercel login).
# A página é estática: o Vercel só entrega o .html; os exames são lidos e calculados no navegador.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="aberrometria-tms"
TEAM="rodrigo-magnos-projects-9b3aaae7"
SITE="https://aberrometria-tms.vercel.app"
VERCEL=(npx --yes vercel@64.5.0)
STAGE="$ROOT/dist_vercel"
TMP=""
trap '[ -n "$TMP" ] && rm -rf "$TMP"' EXIT

if [ $# -ge 1 ]; then
  SRC="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
else
  cd "$ROOT"
  SHA="$(git rev-parse HEAD)"
  RUN="$(gh run list --workflow build.yml --commit "$SHA" --status success --limit 1 --json databaseId --jq '.[0].databaseId // empty')"
  if [ -z "$RUN" ]; then
    echo "Nenhum build do CI concluído com sucesso para $SHA. Faça push e aguarde o CI (ou passe um .html)." >&2
    exit 1
  fi
  git diff --quiet HEAD || echo "Aviso: há alterações não commitadas; será publicado o HTML do commit $SHA (CI)." >&2
  TMP="$(mktemp -d)"
  gh run download "$RUN" --name AberrometriaTMS-html --dir "$TMP"
  SRC="$(ls "$TMP"/AberrometriaTMS-*.html)"
fi
[ -f "$SRC" ] || { echo "Arquivo não encontrado: $SRC" >&2; exit 1; }

mkdir -p "$STAGE"
cp "$SRC" "$STAGE/index.html"
cp "$ROOT/web/vercel.json" "$STAGE/vercel.json"
# Só a página e a configuração sobem para o Vercel (lista de permissão).
printf '/*\n!/index.html\n!/vercel.json\n' > "$STAGE/.vercelignore"
[ -f "$STAGE/.vercel/project.json" ] || "${VERCEL[@]}" link --yes --team "$TEAM" --project "$PROJECT" --cwd "$STAGE"
# O "link" baixa um token temporário (VERCEL_OIDC_TOKEN) em .env.local; a página estática não usa variáveis.
rm -f "$STAGE"/.env*
"${VERCEL[@]}" deploy --prod --yes --cwd "$STAGE"

# Confere se o endereço público entrega exatamente o arquivo publicado.
want="$(shasum -a 256 "$STAGE/index.html" | cut -d' ' -f1)"
got="$(curl -fsSL "$SITE/" | shasum -a 256 | cut -d' ' -f1)"
if [ "$want" = "$got" ]; then
  echo "OK: $SITE entrega $(basename "$SRC") (sha256 $want)"
else
  echo "ATENÇÃO: $SITE não entrega o arquivo publicado (esperado $want, recebido $got)" >&2
  exit 1
fi
