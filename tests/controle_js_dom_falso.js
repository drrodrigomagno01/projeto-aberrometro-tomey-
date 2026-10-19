// DOM falso mínimo para executar o controle de diâmetro do site (tmsab.web.CONTROLE_JS, kcBind) no Node, sem
// navegador — usado por tests/test_controle_diametro_web.py.
//
//   node controle_js_dom_falso.js <controle.js> <dados.json> <passos.json>
//
// <dados.json> = bloco JSON .kc-dados gerado por tmsab.web._controle_html. <passos.json> = lista de passos, cada um
// {"d": "3.0"} (controle deslizante) ou {"ponteiro": [["pointerdown", px, py], ["pointermove", px, py], ...]} (mapa).
// A sobreposição tem 440 × 440 px na tela, com a origem no canto superior esquerdo. Imprime em stdout um JSON com o
// estado da sobreposição e da tabela depois de cada passo.
const fs = require("fs");
const [, , jsFile, dadosFile, passosFile] = process.argv;

class El {
  constructor(tag, ns) { this.tagName = tag; this.ns = ns || null; this.attrs = {}; this.children = []; this._text = ""; this.listeners = {}; this.parent = null; this.className = ""; this.value = ""; }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === "class") this.className = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  appendChild(c) { c.parent = this; this.children.push(c); return c; }
  removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); c.parent = null; return c; }
  get firstChild() { return this.children[0] || null; }
  set textContent(t) { this._text = String(t); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  fire(t, ev) { for (const f of this.listeners[t] || []) f(ev || {}); }
  getBoundingClientRect() { return { width: 440, height: 440, left: 0, top: 0 }; }
  focus() {}
  setPointerCapture() {}
  _classes() { return (this.attrs["class"] || this.className || "").split(/\s+/).filter(Boolean); }
  _simple(s) {                       // tag?(.classe)*
    const m = /^([a-z0-9]*)((?:\.[A-Za-z0-9_-]+)*)$/.exec(s);
    if (!m) throw new Error("seletor não suportado: " + s);
    if (m[1] && m[1] !== this.tagName) return false;
    return m[2].split(".").filter(Boolean).every((c) => this._classes().includes(c));
  }
  matches(sel) { return this._simple(sel); }
  _all(sel, out) {
    const partes = sel.trim().split(/\s+/);
    const walk = (no, i) => {          // casa partes[i..] entre os descendentes de `no`
      for (const c of no.children) {
        let prox = i;
        if (c._simple(partes[i])) { if (i === partes.length - 1) out.push(c); else prox = i + 1; }
        walk(c, prox);
      }
    };
    walk(this, 0);
    return out;
  }
  querySelectorAll(sel) { return this._all(sel, []); }
  querySelector(sel) { return this._all(sel, [])[0] || null; }
}

const raiz = new El("html");
const document = {
  createElementNS: (ns, tag) => new El(tag, ns),
  createElement: (tag) => new El(tag),
  querySelectorAll: (sel) => raiz.querySelectorAll(sel),
  querySelector: (sel) => raiz.querySelector(sel),
  matches: () => false,
};
const window = { console };

const textoDados = fs.readFileSync(dadosFile, "utf8");
const dados = JSON.parse(textoDados);
const L = dados.mapa.L;

// o bloco .kc como tmsab.web._controle_html o gera
const kc = new El("div"); kc.setAttribute("class", "kc");
const inp = new El("input"); inp.setAttribute("class", "kc-d"); inp.setAttribute("type", "range");
inp.setAttribute("value", dados.inicial.toFixed(1)); inp.value = dados.inicial.toFixed(1);
const saida = new El("output"); saida.setAttribute("class", "kc-valor");
const tab = new El("table"); tab.setAttribute("class", "metricas kerato kc-tabela");
const tbody = new El("tbody"); tab.appendChild(tbody);
const svg = new El("svg"); svg.setAttribute("class", "kc-sobre");
const sc = new El("script"); sc.setAttribute("class", "kc-dados"); sc.textContent = textoDados;
// tela → viewBox: 440 px de largura e altura para 2L mm, y da tela para baixo (como no navegador)
svg.getScreenCTM = () => ({ inverse: () => ({ a: 2 * L / 440, b: 0, c: 0, d: 2 * L / 440, e: -L, f: -L }) });
for (const e of [inp, saida, tab, svg, sc]) kc.appendChild(e);
raiz.appendChild(kc);

const codigo = fs.readFileSync(jsFile, "utf8");
const kcBind = new Function("window", "document", codigo + "\n;return kcBind;")(window, document);
kcBind(raiz);
kcBind(raiz);                                   // religar é inofensivo (WeakSet)

function estado() {
  const g = svg.children.find((c) => c.tagName === "g");
  const filhos = g ? g.children : [];
  const linha = (cls) => {
    const l = filhos.find((c) => c.attrs["class"] === cls);
    return l ? { xy: ["x1", "y1", "x2", "y2"].map((k) => parseFloat(l.attrs[k])), cor: l.attrs.stroke,
                 tracejado: l.getAttribute("stroke-dasharray") } : null;
  };
  const circ = filhos.find((c) => c.attrs["class"] === "kc-circulo");
  const rotulo = (n) => {
    const t = svg.children.find((c) => c.attrs["class"] === "kc-rotulo kc-rotulo-" + n);
    return t ? { texto: t.textContent, cor: t.attrs.fill } : null;
  };
  return {
    saida: saida.textContent, valor: inp.value, valorAttr: inp.getAttribute("value"),
    transform: g ? g.attrs.transform : null,
    r_circulos: filhos.filter((c) => c.tagName === "circle").map((c) => parseFloat(c.attrs.r)),
    circulo: circ ? { cor: circ.attrs.stroke, tracejado: circ.getAttribute("stroke-dasharray") } : null,
    k1: linha("kc-k1"), k2: linha("kc-k2"), rot_k1: rotulo("k1"), rot_k2: rotulo("k2"),
    linhas: tbody.children.map((tr) => tr.children.map((td) => td.textContent)),
    classes: tbody.children.map((tr) => tr.className || ""),
  };
}

const res = { L, inicial: estado(), passos: [] };
for (const p of JSON.parse(fs.readFileSync(passosFile, "utf8"))) {
  if (p.d !== undefined) {
    inp.value = p.d;
    inp.fire("input", {});
  } else {
    for (const [tipo, px, py] of p.ponteiro) {
      svg.fire(tipo, { pointerType: "mouse", button: 0, pointerId: 1, clientX: px, clientY: py, preventDefault() {} });
    }
  }
  res.passos.push(estado());
}
process.stdout.write(JSON.stringify(res));
