/* SIRIUS Unified Frontend: Chemistry, 3D Engine, Routing, and All Views */
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
const RM = matchMedia('(prefers-reduced-motion:reduce)').matches;
const dpr = Math.min(devicePixelRatio, 2);

/* ========== 1. Chemistry Layer ========== */
const ez = x => x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x);
const lr = (a, b, t) => a + (b - a) * t;
let sd = 1;
const rnd = () => (sd = (Math.imul(sd, 1664525) + 1013904223) >>> 0) / 4294967296;
const hash = s => { let h = 2166136261; for (const c of s) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return h >>> 0; };
const nz = P => {
  const n = P.length, c = [0, 1, 2].map(k => P.reduce((a, p) => a + p[k], 0) / n);
  const Q = P.map(p => p.map((v, k) => v - c[k])), r = Math.max(...Q.map(p => Math.hypot(...p))) || 1;
  return Q.map(p => p.map(v => v * 3.2 / r));
};

function genP(seed) {
  sd = seed;
  const P = [], V = () => new THREE.Vector3(rnd() - .5, rnd() - .5, rnd() - .5);
  let p = new THREE.Vector3(), d = V().normalize();
  for (let s = 0; s < 10; s++) {
    const u = new THREE.Vector3().crossVectors(d, V()).normalize(), v = new THREE.Vector3().crossVectors(d, u);
    for (let i = 0; i < 24; i++) {
      const a = i * .62;
      P.push(p.clone().addScaledVector(d, i * .2).addScaledVector(u, Math.cos(a) * .7).addScaledVector(v, Math.sin(a) * .7).toArray());
    }
    p.addScaledVector(d, 4.8);
    const e = V().sub(p.clone().multiplyScalar(.15)).normalize();
    for (let i = 0; i < 7; i++) { d.lerp(e, .35).normalize(); p.addScaledVector(d, .45); P.push(p.toArray()); }
    d = e;
  }
  return nz(P);
}

function molModel(seed) {
  sd = seed;
  const at = [], bd = [], r3 = Math.sqrt(3), A = (x, y, z, e) => at.push({ x, y, z, e });
  for (let k = 0; k < 6; k++) { const a = Math.PI / 6 + k * Math.PI / 3; A(Math.cos(a), Math.sin(a), 0, 'C'); }
  [30, 90, 270, 330].forEach(g => A(r3 + Math.cos(g * Math.PI / 180), Math.sin(g * Math.PI / 180), 0, 'C'));
  for (let i = 0; i < 6; i++) bd.push([i, (i + 1) % 6]);
  [[6, 7], [7, 0], [5, 8], [8, 9], [9, 6]].forEach(b => bd.push(b));
  at.forEach(a => { if (rnd() < .22) a.e = 'N'; });
  const pool = [1, 2, 3, 4, 6, 7, 8, 9].sort(() => rnd() - .5).slice(0, 3), E = ['O', 'N', 'F', 'C', 'Cl', 'N'], subs = [];
  pool.forEach(b => {
    const c = b < 6 ? [0, 0] : [r3, 0], o = at[b], dx = o.x - c[0], dy = o.y - c[1], l = Math.hypot(dx, dy);
    const ux = dx / l, uy = dy / l, e = E[Math.floor(rnd() * 6)], i = at.length;
    A(o.x + ux, o.y + uy, (rnd() - .5) * .6, e); bd.push([b, i]); subs.push(e);
    if (e != 'F' && e != 'Cl' && rnd() < .55) {
      const t = (rnd() < .5 ? 1 : -1) * Math.PI / 3, c2 = Math.cos(t), s2 = Math.sin(t);
      A(at[i].x + ux * c2 - uy * s2, at[i].y + ux * s2 + uy * c2, (rnd() - .5) * .6, rnd() < .5 ? 'C' : 'O');
      bd.push([i, i + 1]);
    }
  });
  return { at, bd, subs };
}

const sv = (m, px = 30, pd = 22) => {
  const xs = m.at.map(a => a.x), ys = m.at.map(a => -a.y), x0 = Math.min(...xs), y0 = Math.min(...ys);
  const W = (Math.max(...xs) - x0) * px + pd * 2, H = (Math.max(...ys) - y0) * px + pd * 2;
  const P = a => [(a.x - x0) * px + pd, (-a.y - y0) * px + pd];
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="2D chemical structure">` +
    m.bd.map(([i, j]) => { const a = P(m.at[i]), b = P(m.at[j]); return `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="currentColor" stroke-width="2"/>`; }).join('') +
    m.at.map(a => { if (a.e == 'C') return ''; const p = P(a); return `<circle cx="${p[0]}" cy="${p[1]}" r="10"/><text class="${a.e}" x="${p[0]}" y="${p[1]}">${a.e}</text>`; }).join('') +
    '</svg>';
};

const SC = ['Quinazoline', 'Naphthyridine', 'Benzimidazole', 'Pyrrolopyrimidine', 'Quinoline', 'Indazole'];
const W2 = { O: 'hydroxy', N: 'amino', F: 'fluoro', Cl: 'chloro', C: 'methyl' };

function makeCands(seed, k) {
  const a = [];
  for (let i = 0; i < 24; i++) {
    const m = molModel(seed + i * 31);
    sd = seed + i * 7;
    const c = { m, mw: 230 + rnd() * 270, qed: .45 + rnd() * .5, sas: 1.8 + rnd() * 3.4, lp: .4 + rnd() * 4.4, dk: -(6 + rnd() * 5), sc: SC[Math.floor(rnd() * 6)] };
    c.d = `${c.sc} core, ${[...new Set(m.subs.map(x => W2[x]))].join(', ')}`;
    c.g = []; c.b = [];
    if (c.qed > .7) c.g.push('Drug-like profile'); else if (c.qed < .6) c.b.push('Below typical drug-likeness');
    if (c.sas < 3) c.g.push('Straightforward to synthesize'); else if (c.sas > 4) c.b.push('Hard to synthesize');
    if (c.dk < -8.5) c.g.push('Strong predicted binding'); else if (c.dk > -7.5) c.b.push('Weak predicted binding');
    if (c.lp > 3.8) c.b.push('High lipophilicity risks poor solubility'); else if (c.lp > 1) c.g.push('Balanced lipophilicity');
    if (c.mw < 400) c.g.push('Low weight leaves room to optimize'); else if (c.mw > 440) c.b.push('Heavy for an oral drug');
    if (!c.g.length) c.g.push('No standout strengths');
    if (!c.b.length) c.b.push('No major flags');
    a.push(c);
  }
  return a.slice(0, k).sort((x, y) => y.qed - x.qed).map((c, i) => (c.n = 'SRS-' + String(i + 1).padStart(3, '0'), c));
}

function parsePDB(t) {
  const P = [];
  t.split('\n').forEach(l => {
    if (l.startsWith('ATOM') && l.substr(12, 4).trim() == 'CA') P.push([+l.substr(30, 8), +l.substr(38, 8), +l.substr(46, 8)]);
  });
  return P.length > 20 ? nz(P) : null;
}

function parseSDF(t) {
  const l = t.split('\n');
  if (l.length < 4) return null;
  const n = +l[3].substr(0, 3), b = +l[3].substr(3, 3);
  if (!n) return null;
  const at = [], bd = [];
  for (let i = 0; i < n; i++) { const s = l[4 + i]; if (!s) return null; at.push({ x: +s.substr(0, 10), y: +s.substr(10, 10), z: +s.substr(20, 10), e: s.substr(31, 3).trim() }); }
  for (let i = 0; i < b; i++) { const s = l[4 + n + i]; if (!s) break; bd.push([+s.substr(0, 3) - 1, +s.substr(3, 3) - 1]); }
  return { at, bd, subs: [] };
}

/* ========== 2. 3D Hologram Engine & AppState ========== */
let R, sc, cam, rot, stage, cur;
let tp = new THREE.Vector3(0, -0.3, 0), ts = 1;
let vy = 0, vx = 0, drag = 0, lx = 0, ly = 0;

const U = { uT: { value: 0 } };
const PU = { uM: { value: 0 }, uF: { value: 1 }, uS: { value: 0.1 }, uX: { value: 0 }, uA: { value: 0 }, uB: { value: 0 }, uFd: { value: 0 }, uR: { value: new THREE.Vector4() }, uTgt: { value: new THREE.Matrix4() } };
const MU = { uM: { value: 0 }, uF: { value: 1 }, uS: { value: 0 } };

const HV = `attribute vec3 aD;uniform float uM;varying vec3 vN,vV,vP;void main(){vec3 p=position+aD*uM;vec4 m=modelViewMatrix*vec4(p,1.);vN=normalMatrix*normal;vV=-m.xyz;vP=p;gl_Position=projectionMatrix*m;}`;
const HF = `uniform vec3 uA,uB;uniform float uT,uO,uF;varying vec3 vN,vV,vP;void main(){float f=pow(1.-abs(dot(normalize(vN),normalize(vV))),2.);
float b=smoothstep(.94,1.,sin(vP.y*18.-uT*.7)*.5+.5);vec3 c=mix(uA,uB,f)+uB*b*.35;gl_FragColor=vec4(c,(.07+f*.9+b*.22)*uO*uF);}`;
const PV = `attribute vec3 aD,aC,aA,aV;attribute float aI,aK,aH;uniform float uM,uZ,uS,uT,uX,uA,uB;uniform vec4 uR;uniform mat4 uTgt;varying float vI,vK,vD,vH,vG;
void main(){vec3 p=position+aD*uM;float h=0.;
if(aH>.5){h=aH<1.5?uR.x:(aH<2.5?uR.y:(aH<3.5?uR.z:uR.w));vec3 ot=normalize(aC+vec3(.001))*1.9;vec3 cl=aC+ot*uA+(p-aC)*(1.-.62*uA);vec3 tg=(uTgt*vec4(aA,1.)).xyz;
float b=clamp(uB*1.3-fract(aI*37.)*.3,0.,1.);b=b*b*(3.-2.*b);p=mix(cl,tg,b)+aV*sin(3.14159*b)*uA*.8;}
vec4 m=modelViewMatrix*vec4(p,1.);float f=mix(aK,1.-aK,uX);float pl=smoothstep(.93,1.,sin(aI*40.-uT*.9)*.5+.5);
vI=aI;vK=f+pl;vD=clamp((-m.z-8.)/8.,0.,1.);vH=h;vG=step(.5,aH);gl_Position=projectionMatrix*m;gl_PointSize=uS*(1.+f*.9+pl*1.2+h*1.6)*uZ/-m.z;}`;
const PF = `uniform float uO,uF,uFd;varying float vI,vK,vD,vH,vG;void main(){float d=length(gl_PointCoord-.5);if(d>.5)discard;
vec3 c=mix(vec3(.36,.2,.6),vec3(.78,.66,.9),vI);c=mix(c,vec3(.95,.91,.86),clamp(vK-.9,0.,1.)*.8);c=mix(c,vec3(1.),smoothstep(.2,0.,d)*.5);
float dm=mix(uF,1.,vH)*(1.-vG*uFd);gl_FragColor=vec4(c,smoothstep(.5,0.,d)*(.3+.7*clamp(vK,0.,1.))*(1.-vD*.65)*uO*dm);}`;
const AD = { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide };

const hm = (a, b, sh) => new THREE.ShaderMaterial({
  uniforms: { uA: { value: new THREE.Color(...a) }, uB: { value: new THREE.Color(...b) }, uT: U.uT, uM: sh.uM, uF: sh.uF, uO: { value: 1 } },
  vertexShader: HV, fragmentShader: HF, ...AD
});

const pm = sh => new THREE.ShaderMaterial({
  uniforms: { uM: sh.uM, uS: sh.uS, uT: U.uT, uF: sh.uF, uX: sh.uX || { value: 0 }, uA: sh.uA || { value: 0 }, uB: sh.uB || { value: 0 }, uFd: sh.uFd || { value: 0 }, uR: sh.uR || { value: new THREE.Vector4() }, uTgt: sh.uTgt || { value: new THREE.Matrix4() }, uZ: { value: 300 * dpr * innerHeight / 900 }, uO: { value: 1 } },
  vertexShader: PV, fragmentShader: PF, ...AD
});

const dl = k => {
  sd = 900 + k * 77;
  const a = [rnd() * 6, rnd() * 6, rnd() * 6, rnd() * 6], m = (.7 + rnd() * .6) * 1.2;
  return u => {
    const w = Math.pow(u, 1.1) * m;
    return [(Math.sin(u * 20 + a[0]) + .5 * Math.sin(u * 55 + a[1])) * w, (Math.sin(u * 24 + a[2]) + .5 * Math.cos(u * 47 + a[3])) * w, Math.sin(u * 17 + a[1] + a[2]) * w];
  };
};

/* Data-driven hologram: density, size and colour come from the real C-alpha trace.
   Burial (neighbours within ~9 A) drives brightness; exposed residues shed a looser outer shell. */
const bpC = new WeakMap();
/* Hologram built from the real C-alpha trace. Also finds four real features from that trace:
   both chain ends, the most buried stretch and the most exposed stretch (by neighbour count). */
function bp(P, k, sh, dense = false) {
  if (!P || P.length < 2) return new THREE.Group();
  const n = P.length, V = P.map(p => new THREE.Vector3(...p)), C = new THREE.CatmullRomCurve3(V);
  const D = k < 0 ? () => [0, 0, 0] : dl(k), st = n > 1500 ? 2 : 1, dn = bpC.get(P) || new Array(n).fill(0);
  if (!bpC.has(P)) { for (let i = 0; i < n; i++) for (let j = 0; j < n; j += st) if (Math.abs(i - j) > 3 && V[i].distanceToSquared(V[j]) < 1.1) dn[i]++; bpC.set(P, dn); }
  const mx = Math.max(...dn, 1);
  const s5 = i => { let s = 0, c = 0; for (let j = Math.max(0, i - 4); j <= Math.min(n - 1, i + 4); j++) { s += dn[j]; c++; } return s / c; };
  const lo = Math.floor(n * .1), hi = Math.max(lo + 1, Math.ceil(n * .9)), wd = Math.max(4, Math.round(n * .03));
  let ic = lo, ie = lo;
  for (let i = lo; i < hi; i++) { const s = s5(i); if (s > s5(ic)) ic = i; if (s < s5(ie)) ie = i; }
  const RGN = [[0, 'N-terminus'], [ic, 'Buried core'], [ie, 'Exposed surface patch'], [n - 1, 'C-terminus']].map(([i, name]) => {
    const cen = new THREE.Vector3(); let c = 0;
    for (let j = Math.max(0, i - wd); j <= Math.min(n - 1, i + wd); j++) { cen.add(V[j]); c++; }
    return { i, name, dn: dn[i], cen: cen.divideScalar(c) };
  });
  const rgOf = i => RGN.findIndex(r => Math.abs(i - r.i) <= wd);
  const pos = [], ad = [], ai = [], ak = [], ah = [], ac = [], hp = [[], [], [], []];
  sd = hash(n + ':' + P[0].join(','));
  const add = (p, u, d, rg) => {
    pos.push(p.x, p.y, p.z); ad.push(...D(u)); ai.push(u); ak.push(d); ah.push(rg + 1);
    if (rg >= 0) { const c = RGN[rg].cen; ac.push(c.x, c.y, c.z); hp[rg].push(ak.length - 1); } else ac.push(0, 0, 0);
  };
  const per = dense ? (n > 1500 ? 5 : 8) : (n > 1500 ? 2 : 4);
  const rad = dense ? 0 : 2;
  for (let i = 0; i < n; i++) {
    const d = dn[i] / mx, rg = rgOf(i);
    for (let s = 0; s < per; s++) { const u = Math.min((i + s / per) / (n - 1), 1); add(C.getPoint(u), u, d, rg); }
    for (let s = 0; s < rad; s++) {
      const o = new THREE.Vector3(rnd() - .5, rnd() - .5, rnd() - .5).normalize().multiplyScalar((.05 + .16 * rnd()) * (1.25 - d));
      add(V[i].clone().add(o), i / (n - 1), d * .5, rg);
    }
  }
  const g = new THREE.BufferGeometry(), F = (a, s) => new THREE.Float32BufferAttribute(a, s), z3 = new Array(ak.length * 3).fill(0);
  g.setAttribute('position', F(pos, 3)); g.setAttribute('aD', F(ad, 3)); g.setAttribute('aI', F(ai, 1)); g.setAttribute('aK', F(ak, 1));
  g.setAttribute('aH', F(ah, 1)); g.setAttribute('aC', F(ac, 3)); g.setAttribute('aA', F(z3.slice(), 3)); g.setAttribute('aV', F(z3.slice(), 3));
  const G = new THREE.Group();
  if (dense) {
    const body = new THREE.Mesh(new THREE.TubeGeometry(C, Math.min(n * 3, 900), .13, 8, false), hm([.28, .14, .48], [.78, .66, .92], sh));
    body.userData.alphaScale = .38;
    G.add(body);
  }
  G.add(new THREE.Points(g, pm(sh))); G.userData = { V, dn, mx, RGN, hp }; return G;
}

const MC = { C: [[.5, .5, .55], [.95, .93, .9]], N: [[.35, .33, .5], [.78, .76, .92]], O: [[.55, .42, .3], [.95, .84, .7]], X: [[.6, .65, .75], [1, 1, 1]] };
const SG = new THREE.SphereGeometry(1, 22, 16);
const CG = new THREE.CylinderGeometry(1, 1, 1, 8, 1, true);
function viewerMoleculeScale(mol) {
  if (!mol?.at?.length) return 1;
  const spans = ['x', 'y', 'z'].map(axis => {
    const values = mol.at.map(atom => Number(atom[axis]) || 0);
    return Math.max(...values) - Math.min(...values);
  });
  return Math.min(1, 5 / Math.max(...spans, 1));
}

function bm(M, sh, s = 1, centerOnBounds = false, display = 'ballstick', monochrome = false) {
  if (!M || !M.at || !M.at.length) return new THREE.Group();
  const n = M.at.length;
  const center = key => {
    const values = M.at.map(atom => atom[key]);
    return centerOnBounds ? (Math.min(...values) + Math.max(...values)) / 2 : values.reduce((sum, value) => sum + value, 0) / n;
  };
  const c = new THREE.Vector3(center('x'), center('y'), center('z'));
  const G = new THREE.Group(), V = a => new THREE.Vector3(a.x, a.y, a.z).sub(c).multiplyScalar(s);
  M.at.forEach((a, atomIndex) => {
    if (display === 'lines') return;
    const q = monochrome ? [[.44, .32, .58], [.86, .75, .95]] : (MC[a.e] || MC.X);
    const o = new THREE.Mesh(SG, hm(q[0], q[1], sh));
    o.position.copy(V(a));
    const radius = display === 'surface' ? .58 : display === 'spheres' ? .46 : (a.e == 'C' ? .3 : .34);
    o.scale.setScalar(radius * s);
    o.userData.moleculePick = { kind: 'atom', index: atomIndex, atom: a };
    G.add(o);
  });
  if (display === 'surface' || display === 'spheres') return G;
  (M.bd || []).forEach(([i, j]) => {
    if (!M.at[i] || !M.at[j]) return;
    const a = V(M.at[i]), b = V(M.at[j]), d = b.clone().sub(a), len = d.length();
    if (len < 1e-5) return;
    const o = new THREE.Mesh(CG, hm([.3, .3, .36], [.8, .8, .86], sh));
    o.position.copy(a).add(b).multiplyScalar(.5); o.scale.set((display === 'lines' ? .026 : .07) * s, len, (display === 'lines' ? .026 : .07) * s);
    o.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
    o.userData.moleculePick = { kind: 'bond', atoms: [i, j], a: M.at[i], b: M.at[j] };
    G.add(o);
  });
  return G;
}

const show = g => { if (!rot) return; rot.clear(); rot.add(g); cur = g; };
const fd = (g, v) => { if (g) g.traverse(o => o.material && o.material.uniforms && o.material.uniforms.uO && (o.material.uniforms.uO.value = v * (o.userData.alphaScale ?? 1))); };
const W = () => innerWidth > 860;
let viewZoom = 1;
const moleculeRaycaster = new THREE.Raycaster();
const moleculePointer = new THREE.Vector2();
const atomLabel = (atom, index) => `${atom.e || 'Atom'}${index + 1}`;
function moleculeHover(e) {
  const tip = $('#tip');
  const previewRoute = route === 'cand' || route === 'ana';
  const viewerRoute = route === 'view' && !!viewerCandidateMol && vMode === 'lig';
  if (!tip || (!previewRoute && !viewerRoute) || !cur) { tip?.classList.remove('on', 'mol-tip'); return; }
  const panel = viewerRoute ? $('#vp')?.getBoundingClientRect() : null;
  const rect = viewerRoute && panel
    ? { left: panel.right + 8, right: innerWidth, top: 84, bottom: innerHeight }
    : moleculePreviewRect();
  if (!rect || e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) {
    tip.classList.remove('on', 'mol-tip'); return;
  }
  moleculePointer.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  moleculeRaycaster.setFromCamera(moleculePointer, cam);
  const hit = moleculeRaycaster.intersectObject(cur, true).find(item => item.object.userData.moleculePick);
  if (!hit) { tip.classList.remove('on', 'mol-tip'); return; }
  const item = hit.object.userData.moleculePick;
  const coordinateUnit = viewerRoute ? 'Å' : '2D depiction units (not Å)';
  if (item.kind === 'atom') {
    const { atom, index } = item;
    tip.innerHTML = `<b>Atom ${index + 1} · ${esc(atom.e || 'Unknown element')}</b><span>${esc(atomLabel(atom, index))} · x ${Number(atom.x).toFixed(2)}, y ${Number(atom.y).toFixed(2)}, z ${Number(atom.z).toFixed(2)} ${coordinateUnit}</span>`;
  } else {
    const [i, j] = item.atoms, distance = Math.hypot(item.a.x-item.b.x, item.a.y-item.b.y, item.a.z-item.b.z);
    tip.innerHTML = `<b>Bond · ${esc(atomLabel(item.a, i))} — ${esc(atomLabel(item.b, j))}</b><span>Length ${distance.toFixed(2)} ${coordinateUnit} · bond order not provided</span>`;
  }
  const tipWidth = Math.min(300, innerWidth - 24), left = Math.max(8, Math.min(e.clientX + 14, innerWidth - tipWidth - 8));
  const top = Math.max(8, Math.min(e.clientY + 14, innerHeight - 70));
  tip.style.transform = `translate(${left}px,${top}px)`;
  tip.classList.add('on', 'mol-tip');
}
const moleculePreviewRect = () => {
  if (route !== 'cand' && route !== 'ana') return null;
  if (!W()) return null;
  const leftPanelWidth = route === 'ana'
    ? Math.min(540, innerWidth * .92)
    : Math.min(680, innerWidth * .515);
  const left = innerWidth * .04 + leftPanelWidth + innerWidth * .025;
  const right = innerWidth * .96;
  const top = 84;
  const bottom = innerHeight - innerWidth * .04;
  return { left, right, top, bottom, width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
};
const candidateViewScale = mol => {
  const xs = mol.at.map(atom => atom.x), ys = mol.at.map(atom => atom.y);
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1);
  return Math.min(1, 4.6 / span);
};
const rightPanelCenterX = () => {
  const rect = moleculePreviewRect();
  const panelCenterPx = rect ? (rect.left + rect.right) / 2 : innerWidth / 2;
  const targetZ = route === 'cand' || route === 'ana' ? -1 : 0;
  const viewHeight = 2 * (cam.position.z - targetZ) * Math.tan(cam.fov * Math.PI / 360);
  return ((panelCenterPx - innerWidth / 2) / innerHeight) * viewHeight;
};

const AppState = {
  getConfig() {
    try { const d = sessionStorage.getItem('sirius_config'); return d ? JSON.parse(d) : { pid: 'SIRIUS', lig: 'C1=CC=CC=C1', k: 12, seed: 7 }; }
    catch (e) { return { pid: 'SIRIUS', lig: 'C1=CC=CC=C1', k: 12, seed: 7 }; }
  },
  setConfig(cfg) { try { sessionStorage.setItem('sirius_config', JSON.stringify(cfg)); } catch (e) {} },
  getCands(k = 12, seed = 7) {
    try {
      const d = sessionStorage.getItem('sirius_cands');
      if (d) {
        const candidates = JSON.parse(d);
        if (Array.isArray(candidates) && candidates.every(c => c.smiles && c.m && c.dk !== undefined)) return candidates;
      }
    } catch (e) {}
    return [];
  },
  setCands(c) { try { sessionStorage.setItem('sirius_cands', JSON.stringify(c)); } catch (e) {} },
  getProtein(seed) {
    try { const c = sessionStorage.getItem('sirius_custom_pdb'); if (c) { const p = parsePDB(c); if (p) return p; } } catch (e) {}
    return genP(seed || hash('SIRIUS'));
  },
  setCustomPDB(t, fn) { try { sessionStorage.setItem('sirius_custom_pdb', t); if (fn) sessionStorage.setItem('sirius_pdb_filename', fn); } catch (e) {} },
  clearCustomPDB() { try { sessionStorage.removeItem('sirius_custom_pdb'); sessionStorage.removeItem('sirius_pdb_filename'); } catch (e) {} },
  getLigand(seed) {
    try { const c = sessionStorage.getItem('sirius_custom_sdf'); if (c) { const p = parseSDF(c); if (p) return p; } } catch (e) {}
    return molModel(seed || 11);
  },
  setCustomSDF(t, fn) { try { sessionStorage.setItem('sirius_custom_sdf', t); if (fn) sessionStorage.setItem('sirius_sdf_filename', fn); } catch (e) {} },
  clearCustomSDF() { try { sessionStorage.removeItem('sirius_custom_sdf'); sessionStorage.removeItem('sirius_sdf_filename'); } catch (e) {} }
};

function init3D() {
  const canvas = $('#c');
  if (!canvas) return;
  R = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  R.setPixelRatio(dpr); R.setClearColor(0, 0);
  sc = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(40, 1, .1, 100);
  cam.position.z = 12;
  sc.add(new THREE.AmbientLight(0xffffff, .35), new THREE.HemisphereLight(0xf0eadf, 0x18181a, .55));
  const key = new THREE.DirectionalLight(0xffffff, .9); key.position.set(4, 6, 10); sc.add(key);

  rot = new THREE.Group();
  stage = new THREE.Group();
  stage.add(rot);
  sc.add(stage);

  const rs = () => { if (!R) return; R.setSize(innerWidth, innerHeight, false); cam.aspect = innerWidth / innerHeight; cam.updateProjectionMatrix(); };
  addEventListener('resize', rs); rs();

  addEventListener('pointerdown', e => {
    if (e.target.closest('.pn,button,input,label,header,a,.rw,.ch,#tb,#ap')) return;
    if (route === 'cand' || route === 'ana') {
      const rect = moleculePreviewRect();
      if (!rect || e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) return;
    }
    drag = 1; lx = e.clientX; ly = e.clientY;
    document.body.classList.add('dragging');
    e.preventDefault();
  });
  addEventListener('pointermove', e => {
    if (drag) {
      vy = (e.clientX - lx) * .005; vx = (e.clientY - ly) * .004;
      rot.rotation.y += vy; rot.rotation.x = Math.max(-1.3, Math.min(1.3, rot.rotation.x + vx));
      lx = e.clientX; ly = e.clientY;
    } else moleculeHover(e);
  });
  const stopDrag = () => { drag = 0; document.body.classList.remove('dragging'); };
  addEventListener('pointerup', stopDrag);
  addEventListener('pointercancel', stopDrag);
  addEventListener('wheel', e => {
    const rect = moleculePreviewRect();
    if (!rect || e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) return;
    e.preventDefault();
    viewZoom = Math.max(.35, Math.min(4, viewZoom * Math.exp(-e.deltaY * .001)));
  }, { passive: false });

  stage.position.copy(tp);
  (function animate() {
    requestAnimationFrame(animate);
    const now = performance.now();
    U.uT.value = now / 1000;
    stage.position.lerp(tp, .07);
    const previewRoute = route === 'cand' || route === 'ana';
    stage.scale.setScalar(lr(stage.scale.x, ts * (previewRoute ? viewZoom : 1), .07));
    if (!drag) { rot.rotation.y += vy; vy = lr(vy, 0, .08); rot.rotation.x += vx; vx = lr(vx, 0, .08); }
    if (window.customTick) window.customTick(now);
    if (previewRoute) {
      R.setScissorTest(false);
      R.clear(true, true, true);
      const rect = moleculePreviewRect();
      if (rect && rect.width > 0 && rect.height > 0) {
        R.setViewport(0, 0, innerWidth, innerHeight);
        R.setScissor(rect.left, innerHeight - rect.bottom, rect.width, rect.height);
        R.setScissorTest(true);
        R.render(sc, cam);
        R.setScissorTest(false);
      }
    } else {
      R.setScissorTest(false);
      R.render(sc, cam);
    }
  })();
}

/* ========== 3. Router & Views ========== */
let route = 'home', busy = 0, selIdx = 0;
const LABELS = { home: 'Home', input: 'Design run', run: 'Running', cand: 'Candidates', ana: 'Analysis', view: 'Viewer' };
const TITLES = { home: 'SIRIUS', input: 'SIRIUS - Set Up a Run', run: 'SIRIUS - Candidate Generation Run', cand: 'SIRIUS - Candidate Outputs', ana: 'SIRIUS - Candidate Analysis', view: 'SIRIUS - Structure Viewer' };

async function go(r) {
  if (busy || (r === route && r !== 'ana')) return;
  enter(r);
}

function enter(r) {
  route = r;
  $('#tip')?.classList.remove('on', 'mol-tip');
  document.body.dataset.r = r;
  document.title = TITLES[r] || 'SIRIUS';
  $$('.pg').forEach(p => p.classList.toggle('on', p.id === r));
  $$('header nav a').forEach(a => a.classList.toggle('on', a.dataset.go === r));
  $('#ttl').style.opacity = r === 'home' ? 1 : 0; $('#ttl').style.transform = '';
  window.customTick = null; cam.position.z = 12; PU.uX.value = 0; document.body.style.removeProperty('--st');
  rot.position.set(0, 0, 0); rot.quaternion.identity(); rot.rotation.set(0, 0, 0); vx = vy = 0;
  if (r === 'cand' || r === 'ana') viewZoom = 1;

  const h = r === 'ana' ? `#ana?id=${selIdx}` : r === 'view' && viewerCandidateIdx >= 0 ? `#view?candidate=${viewerCandidateIdx}` : `#${r}`;
  if (location.hash !== h) history.replaceState(null, '', h);

  if (r === 'home') enterHome();
  else if (r === 'input') enterInput();
  else if (r === 'run') enterRun();
  else if (r === 'cand') enterCand();
  else if (r === 'ana') enterAna(selIdx);
  else if (r === 'view') enterViewer();
}

/* --- Home: centered ivory-to-purple SIRIUS title + left/right sliding scroll texts + particle protein --- */
let homeCands = null;
function enterHome() {
  const P = AppState.getProtein(), home = $('#home'), tip = $('#tip'), ttl = $('#ttl'), K = 10;
  const PG = bp(P, -1, PU), { V, dn, mx } = PG.userData, n = V.length;
  homeCands = homeCands || makeCands(7, K);
  const RG = new THREE.Group(), ring = new THREE.Group(), mols = homeCands.map(c => bm(c.m, MU, .4));
  mols.forEach(g => { g.scale.setScalar(.001); fd(g, 0); ring.add(g); });
  const mk = new THREE.Mesh(new THREE.SphereGeometry(.13, 12, 8), new THREE.MeshBasicMaterial({ color: 0xF1E9DC, wireframe: true, transparent: true, opacity: 0 }));
  RG.add(PG, ring, mk); PU.uM.value = 0; PU.uF.value = 1; PU.uS.value = .1; PU.uX.value = 0; show(RG);
  const KY = [0, 1.6, 3.2], KX = [.08, .22, .05], KS = [.76, .88, .72], KZ = [12.5, 11, 12.5], KO = [0, 3.1, -3.1], KOY = [-.35, 0, 0];
  const l3 = (A, p) => p < .5 ? lr(A[0], A[1], ez(p * 2)) : lr(A[1], A[2], ez((p - .5) * 2));
  const mxs = () => Math.max(1, home.scrollHeight - home.clientHeight), v3 = new THREE.Vector3(), secs = $$('#home .sx');
  let sm = 0, uy = 0, ux = 0, amb = 0, last = performance.now(), dragging = 0, px = 0, py = 0, moved = 0, hov = -1, pin = -1, trav = 0;
  home.scrollTop = 0;
  home.onpointerdown = e => { if (e.target.closest('button,a')) return; dragging = 1; trav = 0; px = e.clientX; py = e.clientY; };
  home.onpointermove = e => { const dx = e.clientX - px, dy = e.clientY - py; px = e.clientX; py = e.clientY; moved = 1;
    if (dragging) { trav += Math.abs(dx) + Math.abs(dy); uy += dx * .006; ux = Math.max(-.7, Math.min(.7, ux + dy * .004)); } };
  home.onpointerup = () => { if (dragging && trav < 6) pin = hov >= 0 && pin !== hov ? hov : -1; dragging = 0; };
  home.onpointercancel = () => { dragging = 0; };
  home.onpointerleave = () => { hov = -1; tip.classList.remove('on'); };
  $('#sc').onclick = () => home.scrollTo({ top: innerHeight, behavior: RM ? 'auto' : 'smooth' });

  window.customTick = now => {
    if (route !== 'home') return;
    const dt = Math.min(.05, (now - last) / 1000); last = now;
    sm += (home.scrollTop / mxs() - sm) * (1 - Math.exp(-dt * 6.5));
    const p = Math.min(1, Math.max(0, sm)), w = W();
    amb += RM ? 0 : dt * .07;
    if (!dragging) { const k = Math.exp(-dt * .8); uy *= k; ux *= k; }
    rot.rotation.set(l3(KX, p) + ux, l3(KY, p) + amb + uy, 0);
    cam.position.z = l3(KZ, p); ts = l3(KS, p) * (w ? 1 : .75); tp.set(w ? l3(KO, p) : 0, w ? l3(KOY, p) : 1.1, 0);
    PU.uX.value = ez((p - .2) / .18) * (1 - ez((p - .65) / .18));
    PU.uS.value = .1 + .04 * ez((p - .15) / .15) * (1 - ez((p - .55) / .15));
    ring.rotation.y = now / 4000;
    mols.forEach((g, i) => {
      const a = ez((p - .52 - i * .018) / .16), th = i / K * 6.2832;
      g.position.set(Math.cos(th) * 4.2, Math.sin(th * 2 + now / 2000) * .35, Math.sin(th) * 4.2);
      g.scale.setScalar(.001 + a); fd(g, a);
    });
    ttl.style.opacity = 1 - ez(p * 4.5); ttl.style.transform = `translateY(${-p * 32}vh) scale(${1 + p * .12})`;
    if (moved && !dragging && p < .75) {
      moved = 0; let b = -1, bd = 900;
      for (let i = 0; i < n; i++) { v3.copy(V[i]).applyMatrix4(RG.matrixWorld).project(cam); const dx = (v3.x * .5 + .5) * innerWidth - px, dy = (-v3.y * .5 + .5) * innerHeight - py, d = dx * dx + dy * dy; if (d < bd) { bd = d; b = i; } }
      hov = b;
    }
    const sel = pin >= 0 ? pin : hov;
    if (sel >= 0 && p < .75) {
      mk.position.copy(V[sel]); mk.material.opacity = pin >= 0 ? 1 : .6;
      tip.textContent = `Residue ${sel + 1} of ${n}, burial ${Math.round(dn[sel] / mx * 100)}% (${dn[sel]} neighbours)${pin >= 0 ? ', pinned' : ''}`;
      tip.style.transform = `translate(${Math.min(px + 14, innerWidth - 280)}px,${py + 14}px)`; tip.classList.add('on');
    } else { mk.material.opacity = 0; tip.classList.remove('on'); }
    secs.forEach(s => { const r = s.getBoundingClientRect(), o = (r.top + r.height / 2 - innerHeight / 2) / innerHeight;
      s.style.setProperty('--o', o.toFixed(3)); s.style.setProperty('--v', Math.max(0, Math.min(1, 1.5 - Math.abs(o) * 1.7)).toFixed(3)); });
  };
}

/* --- Input --- */
let customPdb = false, customSdf = false, inputBound = false, selectedComplex = null, activeRunId = null, latestRunArtifacts = null;
function enterInput() {
  const cfg = AppState.getConfig();
  let k = cfg.k || 12;
  const w = W(), P = AppState.getProtein();
  const hasProtein = Boolean(sessionStorage.getItem('sirius_custom_pdb') || (cfg.pid && cfg.pid !== 'SIRIUS' && cfg.pid.trim().length >= 4));
  PU.uM.value = 0; PU.uF.value = hasProtein ? 1.15 : 1; PU.uS.value = hasProtein ? .15 : .1;
  show(bp(P, -1, PU, hasProtein));
  tp.set(w ? 3.7 : 0, 0, w ? -1 : -5);
  ts = w ? .85 : .6;
  fd(cur, w ? 1 : .35);
  window.customTick = () => { if (route === 'input' && !drag && !RM) rot.rotation.y += .0022; };

  $$('#ks button').forEach(b => b.classList.toggle('on', +b.dataset.k === k));

  if (inputBound) return;
  inputBound = true;

  $$('#ks button').forEach(b => b.addEventListener('click', () => {
    k = +b.dataset.k;
    $$('#ks button').forEach(x => x.classList.toggle('on', x === b));
    const c = AppState.getConfig(); c.k = k; AppState.setConfig(c);
  }));

    $('#pf').onchange = e => {
      selectedComplex = e.target.files[0] || null;
      $('#pfn').textContent = selectedComplex ? `${selectedComplex.name} · ${(selectedComplex.size / 1024 / 1024).toFixed(2)} MB` : 'The bound ligand and pocket center are detected automatically.';
      $('#err').textContent = '';
    };
    for (const [id, out] of [['states', 'statesOut'], ['exhaustiveness', 'exhOut']]) {
      const input = $('#' + id), output = $('#' + out);
      input.oninput = () => output.value = input.value;
    }
    $('#go').onclick = async () => {
      const err = $('#err'), button = $('#go');
      if (!selectedComplex) { err.textContent = 'Choose a complete protein-ligand complex first.'; return; }
      err.textContent = ''; button.disabled = true; button.textContent = 'Preparing run…';
      const data = new FormData();
      data.append('complex', selectedComplex, selectedComplex.name);
      data.append('molecules', (document.querySelector('#ks button.on') || { dataset: { k: 6 } }).dataset.k);
      data.append('states', $('#states').value);
      data.append('exhaustiveness', $('#exhaustiveness').value);
      data.append('device', 'cuda');
      try {
        const response = await fetch('/api/runs', { method: 'POST', body: data });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'Could not start the pipeline.');
        activeRunId = result.id;
        $('#pfn').textContent = `${selectedComplex.name} · detected ligand: ${result.ligand} · pocket (${result.center.map(v => Number(v).toFixed(2)).join(', ')}) Å`;
        AppState.setConfig({ k: Number(data.get('molecules')), seed: Date.now(), runId: activeRunId });
        await go('run');
      } catch (error) {
        err.textContent = error.message;
      } finally {
        button.disabled = false; button.textContent = 'Generate candidates';
      }
    };
}

/* --- Run --- */
function asamTraceSVG(coords) {
  if (!Array.isArray(coords) || coords.length < 2) return '';
  const projected = coords.map(([x, y, z]) => [x * .82 - y * .48, z * .84 - x * .36 - y * .62]);
  const xs = projected.map(p => p[0]), ys = projected.map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const scale = Math.min(204 / (maxX - minX || 1), 84 / (maxY - minY || 1));
  const points = projected.map(([x, y]) => `${(110 + (x - (minX + maxX) / 2) * scale).toFixed(1)},${(49 - (y - (minY + maxY) / 2) * scale).toFixed(1)}`).join(' ');
  const markers = projected.filter((_, i) => i % Math.max(1, Math.floor(projected.length / 14)) === 0).map(([x, y]) =>
    `<circle cx="${(110 + (x - (minX + maxX) / 2) * scale).toFixed(1)}" cy="${(49 - (y - (minY + maxY) / 2) * scale).toFixed(1)}" r="1.6"/>`).join('');
  return `<svg viewBox="0 0 220 98" role="img" aria-label="Projected alpha-carbon trace of an aSAM protein state"><polyline points="${points}"/>${markers}</svg>`;
}

function renderASAMStates(states) {
  const grid = $('#run-state-grid'), count = $('#run-state-count'), note = $('#run-state-note');
  if (!grid || !count || !note) return;
  if (!states || !states.length) {
    const P = AppState.getProtein();
    const n = Math.max(2, Math.min(8, parseInt($('#states')?.value || '2', 10) || 2));
    const preview = [];
    for (let s = 1; s <= n; s++) {
      const coords = P.map(([x, y, z], i) => [
        x + Math.sin(s * 1.5 + i * 0.14) * 1.6,
        y + Math.cos(s * 1.2 + i * 0.17) * 1.6,
        z + Math.sin(s * 0.8 + i * 0.21) * 1.6
      ]);
      preview.push({ name: `State ${s} (sampling)`, atoms: coords.length, coords });
    }
    count.textContent = `Sampling ${n} conformational states...`;
    note.textContent = 'aSAM is relaxing the protein backbone into distinct breathing pocket conformations.';
    grid.innerHTML = preview.map(state => `<article class="run-state-card active-pulse"><div><b>${esc(state.name)}</b><small>${Number(state.atoms) || state.coords.length} Cα points</small></div>${asamTraceSVG(state.coords)}</article>`).join('');
    return;
  }
  count.textContent = `${states.length} state${states.length === 1 ? '' : 's'} from aSAM`;
  note.textContent = 'Projected from authentic aSAM-generated Cα breathing coordinates.';
  grid.innerHTML = states.map(state => `<article class="run-state-card"><div><b>${esc(state.name)}</b><small>${Number(state.atoms) || state.coords.length} Cα points</small></div>${asamTraceSVG(state.coords)}</article>`).join('');
}

function enterRun() {
  const status = $('#rs'), detail = $('#rp'), progress = $('#rb'), percent = $('#pc');
  const stages = $$('#rl li');
  $('#skip').hidden = true;
  status.textContent = 'Starting CMD-Gen';
  detail.textContent = 'The uploaded complex has been validated. Waiting for the model pipeline.';
  progress.style.width = '3%'; percent.textContent = '003%';
  stages.forEach((item, index) => { item.className = index === 0 ? 'now' : ''; });
  renderASAMStates([]);
  let stopped = false;
  const poll = async () => {
    if (stopped || route !== 'run' || !activeRunId) return;
    try {
      const response = await fetch(`/api/runs/${activeRunId}`, { cache: 'no-store' });
      const run = await response.json();
      if (!response.ok) throw new Error(run.error || 'Could not read run status.');
      renderASAMStates(run.states || []);
      status.textContent = run.phase || 'Running CMD-Gen';
      detail.textContent = run.status === 'error' ? (run.error || 'The pipeline stopped. See the run log below.') : `Ligand ${run.ligand || ''}; ${run.progress || 0}% complete`;
      progress.style.width = `${run.progress || 0}%`; percent.textContent = `${String(run.progress || 0).padStart(3, '0')}%`;
      stages.forEach((item, index) => { item.className = index < run.stage ? 'on' : index === run.stage && run.status === 'running' ? 'now' : ''; });
      if (run.log && run.log.length) detail.textContent = `${run.phase || 'Running'} · ${run.log[run.log.length - 1]}`;
      if (run.status === 'complete') {
        stopped = true;
        latestRunArtifacts = { report: run.report_url, sdf: run.sdf_url };
        AppState.setCands(run.candidates);
        const download = $('#skip'); download.href = run.report_url; download.hidden = false; download.textContent = 'Download pipeline report';
        $('#rp').textContent = `${run.candidates.length} valid generated molecules were docked and ranked across the receptor states.`;
        setTimeout(() => go('cand'), 1800);
        return;
      }
      if (run.status === 'error') {
        stopped = true;
        detail.textContent = `${run.error || 'The run failed.'} ${run.log?.slice(-4).join(' ') || ''}`;
        return;
      }
    } catch (error) {
      stopped = true; detail.textContent = `Could not reach the local model server: ${error.message}`;
    }
    setTimeout(poll, 1800);
  };
  poll();

  const cfg = AppState.getConfig(), K = cfg.k || 12, seed = cfg.seed || 7, P = AppState.getProtein(seed);
  const visH = () => 2 * cam.position.z * Math.tan(cam.fov * Math.PI / 360), visW = () => visH() * cam.aspect;
  const TOP = 3.5, BOT = 0.45, COL = [0x7770A9, 0x9691C0, 0xF0EADF], GREY = 0x2a2a2e;

  function grid(n, cols) {
    const rows = Math.ceil(n / cols), cs = Math.min(visW() * .92 / cols, (TOP - BOT) / rows), pos = [];
    for (let i = 0; i < n; i++) {
      const c = i % cols, r = Math.floor(i / cols);
      pos.push(new THREE.Vector3((c - (cols - 1) / 2) * cs, (TOP + BOT) / 2 + ((rows - 1) / 2 - r) * cs, 0));
    }
    return { pos, cs };
  }
  const box = s => new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.PlaneGeometry(s * .94, s * .94)),
    new THREE.LineBasicMaterial({ color: GREY, transparent: true, opacity: 0 })
  );

  const wide = visW() / visH() > 1.2, SG2 = grid(10, wide ? 5 : 2), MG = grid(24, wide ? 8 : 4);
  const RG = new THREE.Group(), sts = [], sb = [], mls = [], mb = [];
  const cl = i => i % 3, order = [...Array(10).keys()].sort((a, b) => cl(a) - cl(b) || a - b), rank = [];
  order.forEach((s, r) => rank[s] = r);

  for (let i = 0; i < 10; i++) { const g = bp(P, i, PU), b = box(SG2.cs); g.scale.setScalar(.001); RG.add(g, b); sts.push(g); sb.push(b); }
  for (let i = 0; i < 24; i++) { const g = bm(molModel(seed + i * 31), MU, .8), b = box(MG.cs); g.scale.setScalar(.001); fd(g, 0); RG.add(g, b); mls.push(g); mb.push(b); }

  PU.uM.value = 0; PU.uF.value = 1; PU.uS.value = .1;
  show(RG); tp.set(0, 0, 0); ts = 1;

  const t0 = performance.now();
  let step = -1, ended = false;
  const STC = ['#7770A9', '#9691C0', '#F0EADF', '#7770A9', '#9691C0'];
  const RP = [
    [0, 'Sampling 10 conformational states', 'Your protein relaxes into ten different poses.'],
    [3.4, 'Clustering the states', 'Similar poses group together and reveal distinct pockets.'],
    [5.8, 'Extracting pharmacophores', 'The backbone fades into the features a ligand must meet.'],
    [8.6, 'Generating molecules', 'The transformer turns each pharmacophore map into candidate ligands.'],
    [12.2, `Keeping the best ${K}`, 'Candidates are filtered on drug-likeness, synthesizability and fit.']
  ];
  function finishRun() {
    if (ended) return; ended = true; window.customTick = null;
    AppState.setCands(makeCands(seed, K));
    go('cand');
  }
  $('#skip').onclick = finishRun;

  window.customTick = now => {
    if (ended || route !== 'run') return;
    stage.rotation.set(0, 0, 0); rot.rotation.set(0, 0, 0);
    const t = (now - t0) / 1000, e2 = ez((t - 3.4) / 1.6), e3 = ez((t - 5.8) / 2.2), e4 = ez((t - 8.6) / 1.2);
    PU.uM.value = ez((t - .6) / 2.4); PU.uF.value = 1 - .88 * e3; PU.uS.value = .1 + .28 * e3;
    const sp = SG2.cs * .82 / 7.6;

    sts.forEach((g, i) => {
      const a = ez((t - i * .12) / .6), p = SG2.pos[i].clone().lerp(SG2.pos[rank[i]], e2);
      g.position.copy(p); g.scale.setScalar(sp * a * (1 - e4)); g.visible = e4 < .98;
      sb[i].position.copy(p); sb[i].material.opacity = a * (1 - e4);
      sb[i].material.color.setHex(GREY).lerp(new THREE.Color(COL[cl(i)]), e2);
    });
    mls.forEach((g, i) => {
      const a = ez((t - 9.6 - i * .04) / .5), keep = i < K ? 1 : 1 - ez((t - 12.4) / .9);
      g.position.copy(MG.pos[i]); g.scale.setScalar(.001 + a * MG.cs * .8 / 3.8); fd(g, a * keep);
      mb[i].position.copy(MG.pos[i]); mb[i].material.opacity = a * keep; mb[i].material.color.setHex(i < K ? COL[0] : GREY);
    });

    const pRun = Math.min(t / 14.2, 1);
    $('#rb').style.width = pRun * 100 + '%';
    if ($('#pc')) $('#pc').textContent = String(Math.floor(pRun * 100)).padStart(3, '0') + '%';
    const E5 = [0, 3.4, 5.8, 8.6, 12.2, 14.2];
    $$('#rl li').forEach((li, i) => li.style.setProperty('--p', Math.max(0, Math.min(1, (t - E5[i]) / (E5[i + 1] - E5[i])))));
    let k = 0; RP.forEach((p, i) => { if (t >= p[0]) k = i; });
    if (k !== step) {
      step = k; document.body.style.setProperty('--st', STC[k]); $('#rs').classList.remove('sw'); void $('#rs').offsetWidth; $('#rs').classList.add('sw'); $('#rs').textContent = RP[k][1]; $('#rp').textContent = RP[k][2];
      $$('#rl li').forEach((li, i) => li.className = i < k ? 'on' : i === k ? 'now' : '');
    }
    if (!activeRunId && t > 14.2) finishRun();
  };
}

/* --- Candidates --- */
let candList = [], activeCandIdx = -1;
  const mt = c => [Number(c.mw).toFixed(1), Number(c.qed).toFixed(2), c.sas == null ? '—' : Number(c.sas).toFixed(1), Number(c.lp).toFixed(1), Number(c.dk).toFixed(2)];

function previewMolecule(i) {
  if (!candList[i]) return;
  activeCandIdx = i;
  const c = candList[i], w = W();
  show(bm(c.m, MU, candidateViewScale(c.m), true));
  tp.set(w ? rightPanelCenterX() : 0, 0, w ? -1 : -3);
  ts = w ? .9 : .7;
  fd(cur, 1);
}

function enterCand() {
  const cfg = AppState.getConfig();
  candList = AppState.getCands(cfg.k || 12, cfg.seed || 7);
  $('#ct').textContent = `${candList.length} candidates`;
    $('#downloads').innerHTML = latestRunArtifacts ? `<a class="fb" href="${latestRunArtifacts.sdf}">Download generated SDF</a> <a class="fb" href="${latestRunArtifacts.report}">Download run report</a>` : '';
    if (!candList.length) {
      $('#grid').innerHTML = '<p>No completed pipeline results are loaded yet. Upload a complex and run CMD-Gen first.</p>';
      return;
    }
  $('#grid').innerHTML = candList.map((c, i) => `
    <button class="rw" data-i="${i}">
        <span><b class="rn">${esc(c.n)}</b><small title="${esc(c.smiles)}">${esc(c.smiles)}</small></span>
      ${sv(c.m, 20, 18)}
      ${mt(c).map(v => `<span class="m">${v}</span>`).join('')}
    </button>
  `).join('');
  previewMolecule(0);
}

/* --- Analysis --- */
const REFS = { bick: 'Bickerton et al., Nat. Chem. 2012, 4:90-98', ertl: 'Ertl & Schuffenhauer, J. Cheminform. 2009, 1:8', kit: 'Kitchen et al., Nat. Rev. Drug Discov. 2004, 3:935-949', hop: 'Hopkins et al., Drug Discov. Today 2004, 9:430-431', glee: 'Gleeson, J. Med. Chem. 2008, 51:817-834', hug: 'Hughes et al., Bioorg. Med. Chem. Lett. 2008, 18:4872-4875', lip: 'Lipinski et al., Adv. Drug Deliv. Rev. 2001, 46:3-26', rit: 'Ritchie & Macdonald, Drug Discov. Today 2009, 14:1011-1020', baell: 'Baell & Holloway, J. Med. Chem. 2010, 53:2719-2740', veb: 'Veber et al., J. Med. Chem. 2002, 45:2615-2623' };
function evidence(c) {
  const q = c.qed, s = c.sas, l = c.lp, w = c.mw, d = c.dk, ha = c.m.at.length, le = -d / ha, np = c.m.at.filter(a => a.e == 'N' || a.e == 'O').length, G = [], B = [];
  const A = (arr, t, tag, cf, x, r) => arr.push({ t, tag, cf, x, r: REFS[r] });
  if (q >= .65) A(G, `Drug-like property balance (QED ${q.toFixed(2)})`, 'Predicted', 70, 'QED folds eight descriptors (MW, logP, donors, acceptors, polar surface, rotatable bonds, aromatic rings, alerts) into one 0-1 desirability fitted to known oral drugs. A value in this range puts the molecule near the property space where approved drugs cluster. It is a prior, not proof of oral exposure.', 'bick');
  else if (q < .6) A(B, `Below typical drug-likeness (QED ${q.toFixed(2)})`, 'Predicted', 70, 'One or more descriptors sit outside the range typical of oral drugs. Check which one (weight, lipophilicity or polarity) drives the score before investing in synthesis.', 'bick');
  if (s < 3.2) A(G, `Likely accessible synthesis (SAS ${s.toFixed(1)})`, 'Predicted', 65, 'The synthetic accessibility score rewards common fragments and penalises rare ring systems and stereocentres. A low score suggests commercially available building blocks, but it does not confirm a feasible route.', 'ertl');
  else if (s > 4) A(B, `Synthesis may be costly (SAS ${s.toFixed(1)})`, 'Predicted', 65, 'Rare fragments or complexity raise the score. Expect a longer route and slower design-make-test cycles; ask a medicinal chemist to review before committing.', 'ertl');
  if (d < -8.5) A(G, `Strong predicted binding (${d.toFixed(1)} kcal/mol)`, 'Predicted', 40, 'The docking score ranks this molecule well against its peers for the target pocket. Scoring functions are better at enrichment than at absolute affinity, so treat the number as a ranking signal and confirm with re-docking, MD or an assay.', 'kit');
  else if (d > -7.5) A(B, `Weak predicted binding (${d.toFixed(1)} kcal/mol)`, 'Predicted', 40, 'The docking score sits at the weak end of this set. Poses may be strained or lack key contacts; inspect the pose before discarding, since scoring error is large.', 'kit');
  if (le >= .3) A(G, `Efficient use of heavy atoms (LE ${le.toFixed(2)})`, 'Predicted', 45, `Ligand efficiency is predicted binding energy per heavy atom (${ha} here). Values near or above 0.3 kcal/mol per atom are a common benchmark for tractable leads. It inherits the uncertainty of the docking score.`, 'hop');
  else if (le < .25) A(B, `Low ligand efficiency (LE ${le.toFixed(2)})`, 'Predicted', 45, 'Binding energy per heavy atom is low, so added size is not paying for itself. Optimisation would likely inflate weight and lipophilicity.', 'hop');
  if (l >= 1 && l <= 3.5) A(G, `Balanced lipophilicity (LogP ${l.toFixed(1)})`, 'Established', 60, 'Moderate logP is associated with better solubility, permeability and lower promiscuity. Lipinski flags logP above 5; later analyses favour values below about 3 for ADMET.', 'glee');
  else if (l > 3.8) A(B, `High lipophilicity (LogP ${l.toFixed(1)})`, 'Established', 65, 'Compounds with logP above about 3 and low polar surface area show higher toxicity rates in preclinical studies, and high logP often costs solubility.', 'hug');
  else if (l < .8) A(B, `Low lipophilicity (LogP ${l.toFixed(1)})`, 'Hypothesis', 40, 'Very polar molecules may cross membranes poorly. Whether this holds depends on transporters and polar surface area, which were not modelled.', 'veb');
  if (w < 400) A(G, `Room to optimise (MW ${w.toFixed(0)})`, 'Established', 60, 'Lower weight leaves headroom to add potency or selectivity groups while staying inside oral drug-like limits (Lipinski: 500 Da).', 'lip');
  else if (w > 440) A(B, `Heavy for an oral drug (MW ${w.toFixed(0)})`, 'Established', 60, 'Absorption odds fall as weight approaches 500 Da, and optimisation usually adds more mass.', 'lip');
  if (np >= 2 && np <= 6) A(G, `Polar anchor points (${np} N/O atoms)`, 'Hypothesis', 45, `The ${c.sc.toLowerCase()} core plus ${np} nitrogen/oxygen atoms offers hydrogen-bond donors and acceptors that could anchor the ligand in a polar pocket. This is a structural inference; it needs the docked pose to confirm.`, 'veb');
  else A(B, `Few directional contacts (${np} N/O atoms)`, 'Hypothesis', 40, 'Few polar atoms means binding may lean on shape and hydrophobic contact, which tends to give lower selectivity.', 'veb');
  A(B, 'Planar aromatic core may stack and aggregate', 'Hypothesis', 40, `Flat fused heteroaromatics such as ${c.sc.toLowerCase()} raise the risk of poor solubility and aggregation at assay concentrations. Higher aromatic ring counts correlate with worse developability.`, 'rit');
  A(B, 'No structural-alert screen has been run', 'Established', 90, 'Every value here is computed, not measured. Screen the structure for PAINS and reactive or toxic alerts before purchase or synthesis, since such motifs cause false positives.', 'baell');
  if (!G.length) A(G, 'No standout strengths', 'Predicted', 50, 'No property clears the strength thresholds used here.', 'bick');
  return { G, B, le, ha };
}
function enterAna(i) {
  {
    const anaConfig = AppState.getConfig();
    candList = AppState.getCands(anaConfig.k || 12, anaConfig.seed || 7);
    if (!candList.length) { $('#ap').innerHTML = '<p>No completed run is loaded. Run the pipeline first.</p>'; return; }
    selIdx = ((i || 0) + candList.length) % candList.length;
    const c = candList[selIdx], w = W();
    history.replaceState(null, '', `#ana?id=${selIdx}`);
    const fmt = value => Number.isFinite(Number(value)) ? Number(value).toFixed(2) : 'Not available';
    $('#ap').innerHTML = `
      <button class="bk" data-go="cand">All candidates</button>
      <h2 class="d">${esc(c.n)}</h2>
      <p>Generated by GCPG and ranked using Vina docking across the aSAM receptor states. Scores are computational predictions.</p>
      <div class="s2">${sv(c.m, 30, 20)}</div>
      <div class="mt">
        <div class="mr"><span>Mean Vina score (kcal/mol)</span><b>${fmt(c.dk)}</b></div>
        <div class="mr"><span>Worst state score (kcal/mol)</span><b>${fmt(c.worst)}</b></div>
        <div class="mr"><span>Score variance</span><b>${fmt(c.variance)}</b></div>
        <div class="mr"><span>Molecular weight (Da)</span><b>${fmt(c.mw)}</b></div>
        <div class="mr"><span>QED</span><b>${fmt(c.qed)}</b></div>
        <div class="mr"><span>RDKit LogP</span><b>${fmt(c.lp)}</b></div>
      </div>
      <h3>SMILES</h3><p class="smiles">${esc(c.smiles)}</p>
      <p class="bd">Docking scores are ranking estimates, not experimental binding affinities or evidence of efficacy. No SAS score is calculated in this interface.</p>
      <button class="view-ligand-btn" id="viewStructureBtn" type="button">Open candidate in structure viewer</button>
      <div class="nv"><button id="prevBtn">Previous</button><button id="nextBtn">Next</button></div>`;
    show(bm(c.m, MU, candidateViewScale(c.m), true)); fd(cur, 1);
    tp.set(w ? rightPanelCenterX() : 0, 0, w ? -1 : -3); ts = w ? .9 : .7;
    $('#viewStructureBtn').onclick = async () => {
      let chosen = c;
      if (!chosen.m3d?.at?.length) {
        const runId = AppState.getConfig().runId;
        if (runId) {
          try {
            const response = await fetch(`/api/runs/${runId}`, { cache: 'no-store' });
            const run = await response.json();
            if (response.ok && Array.isArray(run.candidates)) {
              AppState.setCands(run.candidates);
              chosen = run.candidates[selIdx] || chosen;
            }
          } catch (error) {}
        }
      }
      if (!chosen.m3d?.at?.length) { $('#vi').textContent = `${chosen.n} has no saved 3D conformer. The app needs a completed run with candidate SDF coordinates.`; return; }
      viewerCandidateMol = chosen.m3d;
      viewerCandidateLabel = chosen.n;
      viewerCandidateIdx = selIdx;
      sessionStorage.setItem('sirius_view_candidate_idx', String(selIdx));
      go('view');
    };
    $('#prevBtn').onclick = () => enterAna(selIdx - 1);
    $('#nextBtn').onclick = () => enterAna(selIdx + 1);
    return;
  }
  const cfg = AppState.getConfig();
  candList = AppState.getCands(cfg.k || 12, cfg.seed || 7);
  selIdx = ((i || 0) + candList.length) % candList.length;
  const c = candList[selIdx], w = W(), ev = evidence(c);
  history.replaceState(null, '', `#ana?id=${selIdx}`);
  const M = [['Mol. weight', c.mw, 150, 550, 200, 450, 0], ['QED', c.qed, 0, 1, .6, 1, 2], ['SAS', c.sas, 1, 8, 1, 3.5, 1], ['LogP', c.lp, -1, 6, 1, 3.5, 1], ['Dock score', c.dk, -12, -4, -12, -8.5, 1], ['Lig. efficiency', ev.le, 0, .6, .3, .6, 2]];
  const bar = ([n, v, a, b, wa, wb, f]) => { const p = x => Math.max(0, Math.min(100, (x - a) / (b - a) * 100)), ok = v >= wa && v <= wb;
    return `<div class="mr"><span>${n}</span><div class="tk"><u style="left:${p(wa)}%;width:${p(wb) - p(wa)}%"></u><i class="${ok ? 'ok' : ''}" style="left:${p(v)}%"></i></div><b>${v.toFixed(f)}</b></div>`; };
  const cfl = x => x >= 70 ? 'High' : x >= 50 ? 'Moderate' : 'Low';
  const row = e => `<div class="ev"><button class="eh" aria-expanded="false"><span>${e.t}</span><span class="tg ${e.tag[0]}">${e.tag}</span><span class="cf" title="Confidence ${e.cf}%"><i style="width:${e.cf}%"></i></span></button><div class="ed"><div><p>${e.x} Confidence: ${cfl(e.cf)} (${e.cf}%).</p><small>${e.r}</small></div></div></div>`;
  $('#ap').innerHTML = `
    <button class="bk" data-go="cand">All candidates</button>
    <h2 class="d">${c.n}</h2>
    <p>${c.d}. The shaded band marks the favourable range for each property.</p>
    <div class="s2">${sv(c.m)}</div>
    <div class="mt">${M.map(bar).join('')}</div>
    <div class="lg"><span class="tg E">Established</span>literature rule<span class="tg P">Predicted</span>computed here<span class="tg H">Hypothesis</span>untested inference</div>
    <div class="sec g"><h3>Strengths<em>${ev.G.length} findings. Select one for evidence.</em></h3>${ev.G.map(row).join('')}</div>
    <div class="sec b"><h3>Risks<em>${ev.B.length} findings</em></h3>${ev.B.map(row).join('')}</div>
    <div class="nv"><button id="prevBtn">Previous</button><button id="nextBtn">Next</button></div>`;
  $('#ap').onclick = e => { const h = e.target.closest('.eh'); if (!h) return; const o = h.parentElement.classList.toggle('open'); h.setAttribute('aria-expanded', o); };
  show(bm(c.m, MU, 1.3)); fd(cur, 1); tp.set(w ? 3.1 : 0, w ? 0 : 2.2, 0); ts = w ? .85 : .7;
  $('#prevBtn').onclick = () => enterAna(selIdx - 1);
  $('#nextBtn').onclick = () => enterAna(selIdx + 1);
}

/* --- Structure Viewer --- */
let viewerInit = false, vM = null, vSel = -1, vMode = 'prot';
let viewerCandidateMol = null, viewerCandidateLabel = '', viewerCandidateIdx = -1;
let candidateMolStyle = 'cartoon', candidateMolMonochrome = false, candidateMolVisible = true;
const vSt = { style: 'cartoon', color: 'ss', lig: true, pocket: false, spin: true };

function enterViewer() {
  const A3 = { ALA:'A',ARG:'R',ASN:'N',ASP:'D',CYS:'C',GLN:'Q',GLU:'E',GLY:'G',HIS:'H',ILE:'I',LEU:'L',LYS:'K',MET:'M',PHE:'F',PRO:'P',SER:'S',THR:'T',TRP:'W',TYR:'Y',VAL:'V' };
  const EC = { C:0xF1E9DC, N:0x9691C0, O:0xB58B6F, S:0xD9C9A8, P:0x8f89b8, F:0xCFC3D6, CL:0xBFAE92 };
  const CH = [0xF0EADF, 0x9691C0, 0x7770A9, 0xD9C9A8, 0x6a6870, 0xCFC3D6, 0xB58B6F, 0x8f89b8];
  const cc = h => new THREE.Color(h);
  const info = t => { $('#vi').textContent = t; };
  const base = () => W() ? [3.4, 0, 0] : [0, 2.6, 0];

  function parse(t) {
    const at = [], rs = [], het = new Map(), hx = [], sh = []; let rk = '';
    for (const l of t.split('\n')) {
      const r = l.substr(0, 6);
      if (r === 'ENDMDL') break;
      if (r === 'HELIX ') { hx.push([l[19], +l.substr(21, 4), +l.substr(33, 4)]); continue; }
      if (r === 'SHEET ') { sh.push([l[21], +l.substr(22, 4), +l.substr(33, 4)]); continue; }
      const A = r === 'ATOM  ', H = r === 'HETATM';
      if (!A && !H) continue;
      if (l[16] !== ' ' && l[16] !== 'A') continue;
      const nm = l.substr(12, 4).trim(), rn = l.substr(17, 3).trim(), ch = l[21], sq = +l.substr(22, 4);
      const a = { x: +l.substr(30, 8), y: +l.substr(38, 8), z: +l.substr(46, 8), b: +l.substr(60, 6) || 0, nm, rn, ch, sq, res: -1 };
      a.e = (l.substr(76, 2).trim() || nm.replace(/[0-9]/g, '').charAt(0)).toUpperCase();
      if (a.e === 'H' || a.e === 'D' || isNaN(a.x + a.y + a.z)) continue;
      if (A) {
        const k = ch + sq + l[26];
        if (k !== rk) { rk = k; rs.push({ ch, sq, rn, b: a.b, ss: 'C', ca: null, pi: 0, n: rs.length }); }
        a.res = rs.length - 1;
        if (nm === 'CA') { rs[a.res].ca = a; rs[a.res].b = a.b; }
      } else {
        if (rn === 'HOH') continue;
        const k = rn + ch + sq;
        if (!het.has(k)) het.set(k, { rn, ch, sq, at: [] });
        het.get(k).at.push(a);
      }
      at.push(a);
    }
    const P = rs.filter(r => r.ca);
    if (P.length < 10) return null;
    P.forEach((r, i) => {
      r.pi = i;
      for (const [c, a, b] of hx) if (c === r.ch && r.sq >= a && r.sq <= b) r.ss = 'H';
      for (const [c, a, b] of sh) if (c === r.ch && r.sq >= a && r.sq <= b) r.ss = 'E';
    });
    const cen = [0, 1, 2].map(k => P.reduce((s, r) => s + r.ca['xyz'[k]], 0) / P.length);
    const rad = Math.max(...P.map(r => Math.hypot(r.ca.x - cen[0], r.ca.y - cen[1], r.ca.z - cen[2]))) || 1;
    const S = 3.8 / rad;
    at.forEach(a => { a.p = new THREE.Vector3((a.x - cen[0]) * S, (a.y - cen[1]) * S, (a.z - cen[2]) * S); });
    P.forEach(r => { r.cp = r.ca.p; });
    const lig = [...het.values()].filter(h => h.at.length >= 6), la = lig.flatMap(h => h.at), pocket = new Set();
    if (la.length) at.forEach(a => {
      if (a.res < 0 || pocket.has(a.res)) return;
      for (const q of la) if ((a.x - q.x) ** 2 + (a.y - q.y) ** 2 + (a.z - q.z) ** 2 < 20.25) { pocket.add(a.res); break; }
    });
    const bs = P.map(r => r.b);
    return { at, rs, P, S, lig, la, pocket, chs: [...new Set(P.map(r => r.ch))], b0: Math.min(...bs), b1: Math.max(...bs) };
  }

  function rc(r) {
    const m = vSt.color;
    if (m === 'ss') return cc(r.ss === 'H' ? 0xF0EADF : r.ss === 'E' ? 0x9691C0 : 0x6b6a70);
    if (m === 'chain') return cc(CH[vM.chs.indexOf(r.ch) % CH.length]);
    if (m === 'res') { const a = A3[r.rn] || 'X'; return cc('AVLIMFWPGC'.includes(a) ? 0xE8DFD0 : 'KR'.includes(a) ? 0x7770A9 : 'DE'.includes(a) ? 0xB58B6F : 0x9691C0); }
    if (m === 'bfac') return cc(0x3b3a52).lerp(cc(0xF0EADF), Math.min(1, Math.max(0, (r.b - vM.b0) / ((vM.b1 - vM.b0) || 1))));
    return cc(0x3b3a52).lerp(cc(0xF0EADF), r.pi / vM.P.length);
  }
  const ac = a => a.e === 'C' ? (a.res >= 0 ? rc(vM.rs[a.res]) : cc(EC.C)) : cc(EC[a.e] || 0xCCCCCC);

  function tube(g, pts, rad, col) {
    const C = new THREE.CatmullRomCurve3(pts), S = 4, N = (pts.length - 1) * S, R2 = 8, fr = C.computeFrenetFrames(N, false);
    const pos = [], cl = [], ix = [];
    for (let i = 0; i <= N; i++) {
      const f = i / S, a = Math.min(Math.floor(f), pts.length - 2), t = f - a, r = lr(rad[a], rad[a + 1], t), c = col[t < .5 ? a : a + 1], p = C.getPoint(i / N);
      for (let j = 0; j < R2; j++) {
        const q = j / R2 * 6.2832, n = fr.normals[i].clone().multiplyScalar(Math.cos(q)).addScaledVector(fr.binormals[i], Math.sin(q));
        pos.push(p.x + n.x * r, p.y + n.y * r, p.z + n.z * r); cl.push(c.r, c.g, c.b);
      }
    }
    for (let i = 0; i < N; i++) for (let j = 0; j < R2; j++) { const a = i * R2 + j, b = i * R2 + (j + 1) % R2; ix.push(a, b, a + R2, b, b + R2, a + R2); }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(cl, 3));
    geo.setIndex(ix); geo.computeVertexNormals();
    g.add(new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ vertexColors: true, side: THREE.DoubleSide, shininess: 45, specular: 0x2a3340 })));
  }
  function cartoon(g) {
    const S = vM.S;
    vM.chs.forEach(ch => {
      let run = [];
      const flush = () => { if (run.length > 1) tube(g, run.map(r => r.cp), run.map(r => S * (r.ss === 'H' ? 1.15 : r.ss === 'E' ? .9 : .38)), run.map(rc)); run = []; };
      vM.P.filter(r => r.ch === ch).forEach(r => {
        const q = run[run.length - 1];
        if (q && Math.hypot(q.ca.x - r.ca.x, q.ca.y - r.ca.y, q.ca.z - r.ca.z) > 5.5) flush();
        run.push(r);
      });
      flush();
    });
  }
  function bonds(L) {
    const m = new Map(), B = [], k = (x, y, z) => x + ',' + y + ',' + z, cl = v => Math.floor(v / 2);
    L.forEach((a, i) => { const kk = k(cl(a.x), cl(a.y), cl(a.z)); if (!m.has(kk)) m.set(kk, []); m.get(kk).push(i); });
    L.forEach((a, i) => {
      for (let dx = -1; dx < 2; dx++) for (let dy = -1; dy < 2; dy++) for (let dz = -1; dz < 2; dz++) {
        const q = m.get(k(cl(a.x) + dx, cl(a.y) + dy, cl(a.z) + dz));
        if (q) for (const j of q) if (j > i) { const b = L[j], d = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z); if (d < 1.95 && d > .5) B.push([i, j]); }
      }
    });
    return B;
  }
  function lines(g, L) {
    const B = bonds(L), pos = [], cl = [];
    B.forEach(([i, j]) => [L[i], L[j]].forEach(a => { const c = ac(a); pos.push(a.p.x, a.p.y, a.p.z); cl.push(c.r, c.g, c.b); }));
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(cl, 3));
    g.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true })));
  }
  function balls(g, L, r, surf) {
    if (L.length > 14000) L = L.filter((a, i) => i % 2 === 0);
    if (!L.length) return;
    const mat = new THREE.MeshLambertMaterial(surf ? { transparent: true, opacity: .5, depthWrite: false } : {});
    const im = new THREE.InstancedMesh(new THREE.SphereGeometry(1, surf ? 8 : 10, surf ? 6 : 8), mat, L.length), m = new THREE.Matrix4();
    L.forEach((a, i) => { m.makeScale(r, r, r).setPosition(a.p); im.setMatrixAt(i, m); im.setColorAt(i, ac(a)); });
    im.instanceColor.needsUpdate = true; g.add(im);
  }
  function sticks(g, L, r) {
    if (!L.length) return;
    balls(g, L, r * 1.5, false);
    const B = bonds(L); if (!B.length) return;
    const cg = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 8, 1), new THREE.MeshLambertMaterial(), B.length);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), Y = new THREE.Vector3(0, 1, 0), s3 = new THREE.Vector3();
    B.forEach(([i, j], n) => {
      const a = L[i].p, b = L[j].p, d = b.clone().sub(a);
      q.setFromUnitVectors(Y, d.clone().normalize());
      m.compose(a.clone().add(b).multiplyScalar(.5), q, s3.set(r, d.length(), r));
      cg.setMatrixAt(n, m); cg.setColorAt(n, ac(L[i]));
    });
    cg.instanceColor.needsUpdate = true; g.add(cg);
  }
  function build() {
    if (!vM) return;
    vMode = 'prot';
    const g = new THREE.Group(), S = vM.S, prot = vM.at.filter(a => a.res >= 0);
    if (vSt.style === 'cartoon') cartoon(g);
    else if (vSt.style === 'lines') lines(g, prot);
    else balls(g, prot, S * (vSt.style === 'surface' ? 2.4 : 1.5), vSt.style === 'surface');
    if (vSt.pocket) sticks(g, vM.at.filter(a => a.res >= 0 && vM.pocket.has(a.res) && a.nm !== 'CA'), S * .22);
    if (vSt.lig) sticks(g, vM.la, S * .3);
    if (vSel >= 0) {
      const mk = new THREE.Mesh(new THREE.SphereGeometry(S * 2.6, 14, 10), new THREE.MeshBasicMaterial({ color: 0xF2F2EE, wireframe: true }));
      mk.position.copy(vM.rs[vSel].cp); g.add(mk);
    }
    show(g);
  }
  function seq() {
    $('#sq').innerHTML = vM.chs.map(ch => `<div class="qr"><b>${ch}</b>${vM.P.filter(r => r.ch === ch).slice(0, 1500).map(r =>
      `<i data-n="${r.n}" class="${r.n === vSel ? 'sel' : ''}" style="color:#${rc(r).getHexString()}" title="${r.rn} ${r.sq}">${A3[r.rn] || 'x'}</i>`).join('')}</div>`).join('');
  }
  function homePos() { const b = base(); rot.rotation.set(0, 0, 0); vx = vy = 0; ts = 1; tp.set(b[0], b[1], b[2]); }
  function focusAt(p, z) { const b = base(); rot.rotation.set(0, 0, 0); vx = vy = 0; ts = z; tp.set(b[0] - p.x * z, b[1] - p.y * z, -p.z * z); }
  function describe(label) {
    const lg = [...new Set(vM.lig.map(h => h.rn))].join(', '), n = vM.P.length, pc = k => Math.round(vM.P.filter(r => r.ss === k).length / n * 100);
    if ($('#vm')) $('#vm').innerHTML = [[n, 'Residues'], [vM.chs.length, 'Chains'], [vM.at.length, 'Atoms'], [pc('H') + '%', 'Helix'], [pc('E') + '%', 'Sheet'], [lg || 'None', 'Ligand']].map(([v, k]) => `<div><b>${v}</b><span>${k}</span></div>`).join('');
    info(`${label}: ${n} residues, ${vM.chs.length} chain${vM.chs.length > 1 ? 's' : ''}, ${vM.at.length} atoms${lg ? '. Ligand: ' + lg : ''}.`);
  }
  function setCandidateViewerControls(active) {
    $('#candidate-view-controls').hidden = !active;
    $('#vc').hidden = active;
    $('#vs [data-s="cartoon"]').textContent = active ? 'Ball & stick' : 'Cartoon';
    $('#vs [data-s="surface"]').textContent = active ? 'Space filling' : 'Surface';
    const pocketButton = $('#vk [data-k="pocket"]'), ligandButton = $('#vk [data-k="lig"]'), focusButton = $('#vk [data-k="focus"]');
    pocketButton.hidden = active;
    ligandButton.textContent = active ? 'Candidate' : 'Ligand';
    focusButton.textContent = active ? 'Fit candidate' : 'Focus ligand';
    if (active) {
      const candidates = AppState.getCands();
      $('#candidate-view-select').innerHTML = candidates.map((candidate, index) => `<option value="${index}">${esc(candidate.n)}</option>`).join('');
      $('#candidate-view-select').value = String(viewerCandidateIdx);
      const extent = $('#candidate-extent');
      if (extent && viewerCandidateMol?.at?.length) {
        const axes = ['x', 'y', 'z'].map(axis => {
          const values = viewerCandidateMol.at.map(atom => Number(atom[axis]));
          return Math.max(...values) - Math.min(...values);
        });
        extent.textContent = `Coordinate extents: ${axes.map(value => value.toFixed(2)).join(' × ')} Å (X × Y × Z)`;
      }
    }
  }
  function renderCandidateMol(resetView = false) {
    if (!viewerCandidateMol) return;
    vMode = 'lig'; candidateMolVisible = true;
    $$('#vm_mode button').forEach(button => button.classList.toggle('on', button.dataset.m === 'lig'));
    setCandidateViewerControls(true);
    if ($('#sq')) $('#sq').style.display = 'none';
    show(bm(viewerCandidateMol, MU, viewerMoleculeScale(viewerCandidateMol), true, candidateMolStyle, candidateMolMonochrome));
    fd(cur, 1);
    if (resetView) homePos();
    const bonds = viewerCandidateMol.bd?.length || 0;
    info(`${viewerCandidateLabel}: ${viewerCandidateMol.at.length} atoms, ${bonds} bonds. Generated 3D conformer (pre-docking); coordinates are in Å. Drag to rotate; scroll to zoom.`);
  }
  function showLigandSDF(L, label, isCandidate = false) {
    vMode = 'lig';
    $$('#vm_mode button').forEach(b => b.classList.toggle('on', b.dataset.m === 'lig'));
    if ($('#sq')) $('#sq').style.display = 'none';
    if (isCandidate) { renderCandidateMol(true); return; }
    setCandidateViewerControls(false);
    show(bm(L, MU, viewerMoleculeScale(L), true)); fd(cur, 1); homePos();
    info(`${label}: ${L.at.length} atoms, ${L.bd.length} bonds (SDF).`);
  }
  function load(text, label) {
    const p = parse(text); if (!p) { info('Could not read atoms from that file.'); return false; }
    vM = p; vSel = -1;
    $$('#vm_mode button').forEach(b => b.classList.toggle('on', b.dataset.m === 'prot'));
    if ($('#sq')) $('#sq').style.display = '';
    build(); seq(); homePos(); describe(label); return true;
  }
  function demo() {
    const P = AppState.getProtein(), d = P.slice(1).reduce((s, p, i) => s + Math.hypot(p[0] - P[i][0], p[1] - P[i][1], p[2] - P[i][2]), 0) / (P.length - 1), k = 3.8 / d;
    const f = v => (v * k).toFixed(3).padStart(8);
    load(P.map((p, i) => 'ATOM  ' + String(i + 1).padStart(5) + '  CA  ALA A' + String(i + 1).padStart(4) + '    ' + f(p[0]) + f(p[1]) + f(p[2]) + '  1.00' + '20.00'.padStart(6) + '           C').join('\n'), 'Demo structure');
  }
  async function fetchId(id) {
    id = id.trim().toUpperCase();
    if (!/^[0-9][A-Z0-9]{3}$/.test(id)) { info('Enter a 4-character PDB ID, like 7ONS.'); return false; }
    info('Fetching ' + id + ' from RCSB...');
    try {
      const r = await fetch('https://files.rcsb.org/download/' + id + '.pdb');
      if (!r.ok) throw new Error(r.status);
      return load(await r.text(), id);
    } catch (e) { info('Could not fetch ' + id + '. Check the ID and your connection, or open a file instead.'); return false; }
  }

  window.customTick = () => { if (route === 'view' && vSt.spin && !drag && !RM) rot.rotation.y += .0022; };

  if (!viewerInit) {
    viewerInit = true;
    const pick = (sel2, attr, fn) => $$(sel2 + ' button').forEach(b => b.addEventListener('click', () => {
      $$(sel2 + ' button').forEach(x => x.classList.toggle('on', x === b)); fn(b.dataset[attr]);
    }));
    pick('#vm_mode', 'm', m => {
      if (m === 'lig') {
        const name = viewerCandidateLabel || sessionStorage.getItem('sirius_sdf_filename') || 'Demo SDF ligand';
        if (viewerCandidateMol) renderCandidateMol(true);
        else showLigandSDF(AppState.getLigand(), name);
      } else {
        setCandidateViewerControls(false);
        if ($('#sq')) $('#sq').style.display = '';
        if (vM) { build(); seq(); homePos(); describe($('#vid').value.trim().toUpperCase() || 'Structure'); }
        else demo();
      }
    });
    pick('#vs', 's', v => {
      if (viewerCandidateMol && vMode === 'lig') { candidateMolStyle = v; renderCandidateMol(false); return; }
      vSt.style = v; $$('#vm_mode button').forEach(b => b.classList.toggle('on', b.dataset.m === 'prot')); if ($('#sq')) $('#sq').style.display = ''; build();
    });
    pick('#vc', 'c', v => { vSt.color = v; $$('#vm_mode button').forEach(b => b.classList.toggle('on', b.dataset.m === 'prot')); if ($('#sq')) $('#sq').style.display = ''; build(); seq(); });
    $$('#vk button').forEach(b => b.addEventListener('click', () => {
      const k = b.dataset.k;
      if (k === 'reset') { homePos(); return; }
      if (k === 'focus') {
        if (viewerCandidateMol && vMode === 'lig') {
          homePos(); vSt.spin = false; $('#vk [data-k=spin]').classList.remove('on');
          info(`${viewerCandidateLabel}: centered and fitted in the viewer.`); return;
        }
        if (vM && vM.la.length) {
          const c = new THREE.Vector3(); vM.la.forEach(a => c.add(a.p));
          focusAt(c.divideScalar(vM.la.length), 2.6); vSt.spin = false; $('#vk [data-k=spin]').classList.remove('on');
        } else info('This structure has no ligand to focus on.');
        return;
      }
      if (k === 'pocket' && viewerCandidateMol && vMode === 'lig') {
        info('Pocket display is available when viewing a protein structure.'); return;
      }
      if (k === 'lig' && viewerCandidateMol && vMode === 'lig') {
        candidateMolVisible = !candidateMolVisible; b.classList.toggle('on', candidateMolVisible); b.setAttribute('aria-pressed', String(candidateMolVisible));
        if (candidateMolVisible) renderCandidateMol(false); else show(new THREE.Group());
        info(candidateMolVisible ? `${viewerCandidateLabel}: candidate molecule shown.` : `${viewerCandidateLabel}: candidate molecule hidden.`); return;
      }
      vSt[k] = !vSt[k]; b.classList.toggle('on', vSt[k]); b.setAttribute('aria-pressed', vSt[k]);
      if (k !== 'spin') build();
    }));
    $('#candidate-view-select').addEventListener('change', e => {
      const index = Number(e.target.value), candidates = AppState.getCands();
      const candidate = candidates[index];
      if (!candidate?.m3d?.at?.length) { info(`${candidate?.n || 'This candidate'} has no 3D conformer. Rerun candidate generation to load its SDF coordinates.`); return; }
      viewerCandidateIdx = index; viewerCandidateMol = candidate.m3d; viewerCandidateLabel = candidate.n;
      sessionStorage.setItem('sirius_view_candidate_idx', String(index));
      history.replaceState(null, '', `#view?candidate=${index}`);
      renderCandidateMol(true);
    });
    $('#candidate-colors').addEventListener('click', e => {
      const button = e.target.closest('button[data-mol-color]'); if (!button) return;
      candidateMolMonochrome = button.dataset.molColor === 'mono';
      $('#candidate-colors').querySelectorAll('button').forEach(item => item.classList.toggle('on', item === button));
      renderCandidateMol(false);
    });
    $('#sq').addEventListener('click', e => {
      const i = e.target.closest('i[data-n]'); if (!i || !vM) return;
      vSel = +i.dataset.n; build(); seq();
      const r = vM.rs[vSel]; focusAt(r.cp, 2.4); info(`${r.rn} ${r.sq}, chain ${r.ch}. B-factor ${r.b.toFixed(1)}.`);
    });
    $('#vg').onclick = () => fetchId($('#vid').value);
    $('#vid').addEventListener('keydown', e => { if (e.key === 'Enter') fetchId(e.target.value); });
    $('#vf').onchange = async e => {
      const f = e.target.files[0]; if (!f) return;
      const t = await f.text();
      if (/\.(sdf|mol)$/i.test(f.name)) {
        let L = null; try { L = parseSDF(t); } catch (x) {}
        if (!L) { info('Could not read that SDF file.'); return; }
        AppState.setCustomSDF(t, f.name);
        showLigandSDF(L, f.name);
      } else if (load(t, f.name)) AppState.setCustomPDB(t, f.name);
    };
    addEventListener('wheel', e => {
      if (route !== 'view' || e.target.closest('.pn')) return;
      ts = Math.min(5, Math.max(.4, ts * Math.exp(-e.deltaY * .001)));
    }, { passive: true });
  }

  if (viewerCandidateIdx < 0) {
    const stored = sessionStorage.getItem('sirius_view_candidate_idx');
    const saved = stored == null ? -1 : Number(stored);
    if (Number.isInteger(saved) && saved >= 0) viewerCandidateIdx = saved;
  }
  if (viewerCandidateIdx >= 0) {
    const candidates = AppState.getCands(), candidate = candidates[viewerCandidateIdx];
    if (candidate?.m3d?.at?.length) { viewerCandidateMol = candidate.m3d; viewerCandidateLabel = candidate.n; }
  }
  if (viewerCandidateMol) {
    showLigandSDF(viewerCandidateMol, viewerCandidateLabel || 'Selected candidate', true);
    return;
  }
  const cfg = AppState.getConfig(), custom = sessionStorage.getItem('sirius_custom_pdb');
  if (custom && load(custom, sessionStorage.getItem('sirius_pdb_filename') || 'Your file')) return;
  const id = /^[0-9][A-Za-z0-9]{3}$/.test(cfg.pid || '') ? cfg.pid : '7ONS';
  $('#vid').value = id.toUpperCase();
  fetchId(id).then(ok => { if (!ok && !vM) demo(); });
}

/* ========== 4. Global Events & Bootstrap ========== */
document.addEventListener('click', e => {
  const g = e.target.closest('[data-go]');
  if (g) { e.preventDefault(); go(g.dataset.go); return; }
  const c = e.target.closest('#grid .rw[data-i]');
  if (c && route === 'cand') { selIdx = +c.dataset.i; go('ana'); }
});

document.addEventListener('mouseover', e => {
  const c = e.target.closest('#grid .rw[data-i]');
  if (c && route === 'cand' && +c.dataset.i !== activeCandIdx) previewMolecule(+c.dataset.i);
});

function parseHash() {
  const raw = location.hash.replace(/^#/, '');
  const [r, q] = raw.split('?');
  if (r === 'ana' && q) {
    const id = parseInt(new URLSearchParams(q).get('id'), 10);
    if (!isNaN(id)) selIdx = id;
  } else if (r === 'view' && q) {
    const id = parseInt(new URLSearchParams(q).get('candidate'), 10);
    if (!isNaN(id) && id >= 0) viewerCandidateIdx = id;
  }
  return LABELS[r] ? r : 'home';
}

window.addEventListener('hashchange', () => {
  const r = parseHash();
  if (r !== route) go(r);
});

window.addEventListener('DOMContentLoaded', () => {
  init3D();
  enter(parseHash());
});
