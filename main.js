/**
 * main.js — JavaScript (Node.js) port of the Java VectorDB Engine
 *
 * Dependencies: Node.js 18+ (uses built-in http, fetch)
 *
 * Run:  node main.js
 *
 * Requires Ollama running locally for RAG endpoints:
 *   https://ollama.com
 *   ollama pull nomic-embed-text
 *   ollama pull llama3.2
 */

'use strict';

const http = require('http');
const url  = require('url');
const fs   = require('fs');
const path = require('path');

const DIMS = 16;

// =====================================================================
//  DISTANCE METRICS
// =====================================================================

function euclidean(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  if (na < 1e-9 || nb < 1e-9) return 1.0;
  return 1.0 - dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function manhattan(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s;
}

function getDistFn(m) {
  if (m === 'cosine')    return cosine;
  if (m === 'manhattan') return manhattan;
  return euclidean;
}

// =====================================================================
//  DATA TYPES
// =====================================================================

class VectorItem {
  constructor(id, metadata, category, emb) {
    this.id       = id;
    this.metadata = metadata;
    this.category = category;
    this.emb      = emb;
  }
}

// =====================================================================
//  BRUTE FORCE
// =====================================================================

class BruteForce {
  constructor() { this.items = []; }

  insert(v) { this.items.push(v); }

  knn(q, k, distFn) {
    const r = this.items.map(v => [distFn(q, v.emb), v.id]);
    r.sort((a, b) => a[0] - b[0]);
    return r.slice(0, k);
  }

  remove(id) {
    this.items = this.items.filter(v => v.id !== id);
  }
}

// =====================================================================
//  KD-TREE
// =====================================================================

class KDNode {
  constructor(item) { this.item = item; this.left = null; this.right = null; }
}

class KDTree {
  constructor(dims) { this.dims = dims; this.root = null; }

  _ins(node, v, d) {
    if (!node) return new KDNode(v);
    const ax = d % this.dims;
    if (v.emb[ax] < node.item.emb[ax]) node.left  = this._ins(node.left,  v, d + 1);
    else                               node.right = this._ins(node.right, v, d + 1);
    return node;
  }

  insert(v) { this.root = this._ins(this.root, v, 0); }

  _knnSearch(node, q, k, d, distFn, heap) {
    if (!node) return;
    const dn = distFn(q, node.item.emb);
    if (heap.length < k || dn < heap[0][0]) {
      heap.push([dn, node.item.id]);
      // Max-heap: keep only k best
      heap.sort((a, b) => b[0] - a[0]);
      if (heap.length > k) heap.shift();
    }
    const ax   = d % this.dims;
    const diff = q[ax] - node.item.emb[ax];
    const closer  = diff < 0 ? node.left  : node.right;
    const farther = diff < 0 ? node.right : node.left;
    this._knnSearch(closer,  q, k, d + 1, distFn, heap);
    if (heap.length < k || Math.abs(diff) < heap[0][0])
      this._knnSearch(farther, q, k, d + 1, distFn, heap);
  }

  knn(q, k, distFn) {
    const heap = [];
    this._knnSearch(this.root, q, k, 0, distFn, heap);
    heap.sort((a, b) => a[0] - b[0]);
    return heap;
  }

  rebuild(items) {
    this.root = null;
    for (const v of items) this.insert(v);
  }
}

// =====================================================================
//  HNSW — Hierarchical Navigable Small World
// =====================================================================

class HNSWNode {
  constructor(item, maxLyr) {
    this.item   = item;
    this.maxLyr = maxLyr;
    this.nbrs   = [];
    for (let i = 0; i <= maxLyr; i++) this.nbrs.push([]);
  }
}

class HNSW {
  constructor(m, efBuild) {
    this.M        = m;
    this.M0       = 2 * m;
    this.efBuild  = efBuild;
    this.mL       = 1.0 / Math.log(m);
    this.G        = new Map();   // id -> HNSWNode
    this.topLayer = -1;
    this.entryPt  = -1;
    this._seed    = 42;
  }

  _rand() {
    // Simple LCG PRNG for reproducibility
    this._seed = (this._seed * 1664525 + 1013904223) & 0xffffffff;
    return ((this._seed >>> 0) / 4294967296);
  }

  _randLevel() {
    return Math.floor(-Math.log(this._rand() || 1e-10) * this.mL);
  }

  _searchLayer(q, ep, ef, lyr, distFn) {
    const vis   = new Set([ep]);
    // Min-heap for candidates: [dist, id]
    const cands = [[distFn(q, this.G.get(ep).item.emb), ep]];
    // Max-heap for found
    const found = [[distFn(q, this.G.get(ep).item.emb), ep]];

    while (cands.length > 0) {
      cands.sort((a, b) => a[0] - b[0]);
      const [cd, cid] = cands.shift();
      found.sort((a, b) => b[0] - a[0]);
      if (found.length >= ef && cd > found[0][0]) break;

      const cn = this.G.get(cid);
      if (!cn || lyr >= cn.nbrs.length) continue;

      for (const nid of cn.nbrs[lyr]) {
        if (vis.has(nid) || !this.G.has(nid)) continue;
        vis.add(nid);
        const nd = distFn(q, this.G.get(nid).item.emb);
        found.sort((a, b) => b[0] - a[0]);
        if (found.length < ef || nd < found[0][0]) {
          cands.push([nd, nid]);
          found.push([nd, nid]);
          if (found.length > ef) {
            found.sort((a, b) => b[0] - a[0]);
            found.shift();
          }
        }
      }
    }

    found.sort((a, b) => a[0] - b[0]);
    return found;
  }

  _selectNbrs(cands, maxM) {
    return cands.slice(0, maxM).map(c => c[1]);
  }

  insert(item, distFn) {
    const id  = item.id;
    const lvl = this._randLevel();
    this.G.set(id, new HNSWNode(item, lvl));

    if (this.entryPt === -1) { this.entryPt = id; this.topLayer = lvl; return; }

    let ep = this.entryPt;
    for (let lc = this.topLayer; lc > lvl; lc--) {
      const epNode = this.G.get(ep);
      if (epNode && lc < epNode.nbrs.length) {
        const W = this._searchLayer(item.emb, ep, 1, lc, distFn);
        if (W.length > 0) ep = W[0][1];
      }
    }

    for (let lc = Math.min(this.topLayer, lvl); lc >= 0; lc--) {
      const W    = this._searchLayer(item.emb, ep, this.efBuild, lc, distFn);
      const maxM = lc === 0 ? this.M0 : this.M;
      const sel  = this._selectNbrs(W, maxM);
      this.G.get(id).nbrs[lc] = [...sel];

      for (const nid of sel) {
        const nn = this.G.get(nid);
        if (!nn) continue;
        while (nn.nbrs.length <= lc) nn.nbrs.push([]);
        const conn = nn.nbrs[lc];
        conn.push(id);
        if (conn.length > maxM) {
          const ds = conn
            .filter(c => this.G.has(c))
            .map(c => [distFn(nn.item.emb, this.G.get(c).item.emb), c]);
          ds.sort((a, b) => a[0] - b[0]);
          const limited = ds.slice(0, maxM).map(d => d[1]);
          conn.length = 0;
          conn.push(...limited);
        }
      }
      if (W.length > 0) ep = W[0][1];
    }

    if (lvl > this.topLayer) { this.topLayer = lvl; this.entryPt = id; }
  }

  knn(q, k, ef, distFn) {
    if (this.entryPt === -1) return [];
    let ep = this.entryPt;
    for (let lc = this.topLayer; lc > 0; lc--) {
      const epNode = this.G.get(ep);
      if (epNode && lc < epNode.nbrs.length) {
        const W = this._searchLayer(q, ep, 1, lc, distFn);
        if (W.length > 0) ep = W[0][1];
      }
    }
    const W = this._searchLayer(q, ep, Math.max(ef, k), 0, distFn);
    return W.slice(0, k);
  }

  remove(id) {
    if (!this.G.has(id)) return;
    for (const nd of this.G.values())
      for (const layer of nd.nbrs)
        for (let i = layer.length - 1; i >= 0; i--)
          if (layer[i] === id) layer.splice(i, 1);
    if (this.entryPt === id) {
      this.entryPt = -1;
      for (const nid of this.G.keys()) if (nid !== id) { this.entryPt = nid; break; }
    }
    this.G.delete(id);
  }

  getInfo() {
    const gi = {
      topLayer: this.topLayer,
      nodeCount: this.G.size,
      nodesPerLayer: [],
      edgesPerLayer: [],
      nodes: [],
      nodesMeta: [],
      nodesCat:  [],
      edges: []
    };
    const maxL = Math.max(this.topLayer + 1, 1);
    gi.nodesPerLayer = new Array(maxL).fill(0);
    gi.edgesPerLayer = new Array(maxL).fill(0);

    for (const [id, nd] of this.G.entries()) {
      gi.nodes.push([id, nd.maxLyr]);
      gi.nodesMeta.push(nd.item.metadata);
      gi.nodesCat.push(nd.item.category);
      for (let lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
        gi.nodesPerLayer[lc]++;
        if (lc < nd.nbrs.length) {
          for (const nid of nd.nbrs[lc]) {
            if (id < nid) {
              gi.edgesPerLayer[lc]++;
              gi.edges.push([id, nid, lc]);
            }
          }
        }
      }
    }
    return gi;
  }

  get size() { return this.G.size; }
}

// =====================================================================
//  VECTOR DATABASE  (demo 16D index)
// =====================================================================

class VectorDB {
  constructor(dims) {
    this.dims   = dims;
    this.store  = new Map();
    this.bf     = new BruteForce();
    this.kdt    = new KDTree(dims);
    this.hnsw   = new HNSW(16, 200);
    this.nextId = 1;
  }

  insert(meta, cat, emb, distFn) {
    const v = new VectorItem(this.nextId++, meta, cat, emb);
    this.store.set(v.id, v);
    this.bf.insert(v);
    this.kdt.insert(v);
    this.hnsw.insert(v, distFn);
    return v.id;
  }

  remove(id) {
    if (!this.store.has(id)) return false;
    this.store.delete(id);
    this.bf.remove(id);
    this.hnsw.remove(id);
    this.kdt.rebuild([...this.store.values()]);
    return true;
  }

  search(q, k, metric, algo) {
    const dfn = getDistFn(metric);
    const t0  = process.hrtime.bigint();
    let raw;
    if      (algo === 'bruteforce') raw = this.bf.knn(q, k, dfn);
    else if (algo === 'kdtree')     raw = this.kdt.knn(q, k, dfn);
    else                            raw = this.hnsw.knn(q, k, 50, dfn);
    const us = Number(process.hrtime.bigint() - t0) / 1000;

    const hits = [];
    for (const r of raw) {
      const id = r[1];
      if (this.store.has(id)) {
        const v = this.store.get(id);
        hits.push({ id, meta: v.metadata, cat: v.category, emb: v.emb, dist: r[0] });
      }
    }
    return { hits, us: Math.round(us), algo, metric };
  }

  benchmark(q, k, metric) {
    const dfn = getDistFn(metric);
    let t;
    t = process.hrtime.bigint(); this.bf.knn(q, k, dfn);   const bfUs   = Number(process.hrtime.bigint() - t) / 1000;
    t = process.hrtime.bigint(); this.kdt.knn(q, k, dfn);  const kdUs   = Number(process.hrtime.bigint() - t) / 1000;
    t = process.hrtime.bigint(); this.hnsw.knn(q, k, 50, dfn); const hnswUs = Number(process.hrtime.bigint() - t) / 1000;
    return { bfUs: Math.round(bfUs), kdUs: Math.round(kdUs), hnswUs: Math.round(hnswUs), n: this.store.size };
  }

  all()      { return [...this.store.values()]; }
  hnswInfo() { return this.hnsw.getInfo(); }
  get size() { return this.store.size; }
}

// =====================================================================
//  TEXT CHUNKER
// =====================================================================

function chunkText(text, chunkWords, overlapWords) {
  const words = text.trim().split(/\s+/);
  if (words.length === 0) return [];
  if (words.length <= chunkWords) return [text];

  const chunks = [];
  const step   = chunkWords - overlapWords;
  for (let i = 0; i < words.length; i += step) {
    const end = Math.min(i + chunkWords, words.length);
    chunks.push(words.slice(i, end).join(' '));
    if (end === words.length) break;
  }
  return chunks;
}

// =====================================================================
//  OLLAMA CLIENT
// =====================================================================

class OllamaClient {
  constructor(host, port) {
    this.host       = host;
    this.port       = port;
    this.embedModel = 'nomic-embed-text';
    this.genModel   = 'llama3.2';
    this.baseUrl    = `http://${host}:${port}`;
  }

  _esc(s) {
    return s
      .replace(/\\/g, '\\\\')
      .replace(/"/g,  '\\"')
      .replace(/\n/g, '\\n')
      .replace(/\r/g, '\\r')
      .replace(/\t/g, '\\t');
  }

  async isAvailable() {
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 2000);
      const res = await fetch(`${this.baseUrl}/api/tags`, { signal: ctrl.signal });
      clearTimeout(timer);
      return res.ok;
    } catch { return false; }
  }

  async embed(text) {
    try {
      const body = JSON.stringify({ model: this.embedModel, prompt: text });
      const res  = await fetch(`${this.baseUrl}/api/embeddings`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        signal: AbortSignal.timeout(30000)
      });
      if (!res.ok) return [];
      const data = await res.json();
      return data.embedding || [];
    } catch { return []; }
  }

  async generate(prompt) {
    try {
      const body = JSON.stringify({ model: this.genModel, prompt, stream: false });
      const res  = await fetch(`${this.baseUrl}/api/generate`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        signal: AbortSignal.timeout(180000)
      });
      if (!res.ok) return 'ERROR: Ollama unavailable. Run: ollama serve';
      const data = await res.json();
      return data.response || '';
    } catch {
      return 'ERROR: Ollama unavailable. Run: ollama serve';
    }
  }
}

// =====================================================================
//  DOCUMENT DATABASE
// =====================================================================

class DocItem {
  constructor(id, title, text, emb) {
    this.id = id; this.title = title; this.text = text; this.emb = emb;
  }
}

class DocumentDB {
  constructor() {
    this.store  = new Map();
    this.hnsw   = new HNSW(16, 200);
    this.bf     = new BruteForce();
    this.nextId = 1;
    this.dims   = 0;
  }

  insert(title, text, emb) {
    if (this.dims === 0) this.dims = emb.length;
    const item = new DocItem(this.nextId++, title, text, emb);
    this.store.set(item.id, item);
    const vi = new VectorItem(item.id, title, 'doc', emb);
    this.hnsw.insert(vi, cosine);
    this.bf.insert(vi);
    return item.id;
  }

  search(q, k, maxDist) {
    if (this.store.size === 0) return [];
    const raw = this.store.size < 10
      ? this.bf.knn(q, k, cosine)
      : this.hnsw.knn(q, k, 50, cosine);
    return raw.filter(r => r[0] <= maxDist);
  }

  remove(id) {
    if (!this.store.has(id)) return false;
    this.store.delete(id);
    this.hnsw.remove(id);
    this.bf.remove(id);
    return true;
  }

  all()      { return [...this.store.values()]; }
  get size() { return this.store.size; }
}

// =====================================================================
//  JSON / PARSE HELPERS
// =====================================================================

function jS(s) {
  if (s == null) return '""';
  return JSON.stringify(String(s));
}

function jVec(v) {
  return '[' + v.map(x => x.toFixed(4)).join(',') + ']';
}

function parseVec(s) {
  if (!s) return [];
  return s.split(',').map(t => parseFloat(t.trim())).filter(n => !isNaN(n));
}

function parseQuery(queryString) {
  if (!queryString) return {};
  const map = {};
  for (const part of queryString.split('&')) {
    const [k, v] = part.split('=');
    if (k && v !== undefined) map[k] = decodeURIComponent(v);
  }
  return map;
}

async function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end',  () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function parseBodyFields(b) {
  try {
    const obj = JSON.parse(b);
    return {
      meta:  obj.metadata || '',
      cat:   obj.category || '',
      emb:   Array.isArray(obj.embedding) ? obj.embedding.map(Number) : [],
      valid: !!(obj.metadata && Array.isArray(obj.embedding) && obj.embedding.length > 0)
    };
  } catch {
    return { meta: '', cat: '', emb: [], valid: false };
  }
}

// =====================================================================
//  HTTP HELPERS
// =====================================================================

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin',  '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

function send(res, status, body) {
  res.setHeader('Content-Type', 'application/json');
  const buf = Buffer.from(body, 'utf8');
  res.writeHead(status, { 'Content-Length': buf.length });
  res.end(buf);
}

// =====================================================================
//  DEMO DATA LOADER
// =====================================================================

function loadDemo(db) {
  const dist = getDistFn('cosine');
  const data = [
    ['Linked List: nodes connected by pointers', 'cs',
      [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]],
    ['Binary Search Tree: O(log n) search and insert', 'cs',
      [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]],
    ['Dynamic Programming: memoization overlapping subproblems', 'cs',
      [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]],
    ['Graph BFS and DFS: breadth and depth first traversal', 'cs',
      [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]],
    ['Hash Table: O(1) lookup with collision chaining', 'cs',
      [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]],
    ['Calculus: derivatives integrals and limits', 'math',
      [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]],
    ['Linear Algebra: matrices eigenvalues eigenvectors', 'math',
      [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]],
    ['Probability: distributions random variables Bayes theorem', 'math',
      [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]],
    ['Number Theory: primes modular arithmetic RSA cryptography', 'math',
      [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]],
    ['Combinatorics: permutations combinations generating functions', 'math',
      [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]],
    ['Neapolitan Pizza: wood-fired dough San Marzano tomatoes', 'food',
      [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]],
    ['Sushi: vinegared rice raw fish and nori rolls', 'food',
      [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]],
    ['Ramen: noodle soup with chashu pork and soft-boiled eggs', 'food',
      [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]],
    ['Tacos: corn tortillas with carnitas salsa and cilantro', 'food',
      [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]],
    ['Croissant: laminated pastry with buttery flaky layers', 'food',
      [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]],
    ['Basketball: fast-paced shooting dribbling slam dunks', 'sports',
      [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]],
    ['Football: tackles touchdowns field goals and strategy', 'sports',
      [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]],
    ['Tennis: racket volleys groundstrokes and Wimbledon serves', 'sports',
      [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]],
    ['Chess: openings endgames tactics strategic board game', 'sports',
      [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]],
    ['Swimming: butterfly freestyle backstroke Olympic competition', 'sports',
      [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]],
  ];
  for (const [meta, cat, emb] of data) db.insert(meta, cat, emb, dist);
}

// =====================================================================
//  MAIN — HTTP SERVER
// =====================================================================

async function main() {
  const db     = new VectorDB(DIMS);
  const docDB  = new DocumentDB();
  const ollama = new OllamaClient('127.0.0.1', 11434);

  loadDemo(db);

  const ollamaUp = await ollama.isAvailable();
  console.log('=== VectorDB Engine ===');
  console.log('http://localhost:8080');
  console.log(`${db.size} demo vectors | ${DIMS} dims | HNSW+KD-Tree+BruteForce`);
  console.log(`Ollama: ${ollamaUp ? 'ONLINE' : 'OFFLINE (install from ollama.com)'}`);
  if (ollamaUp) console.log(`  embed model: ${ollama.embedModel}  gen model: ${ollama.genModel}`);

  const server = http.createServer(async (req, res) => {
    cors(res);

    const parsed   = url.parse(req.url, true);
    const pathname = parsed.pathname;
    const method   = req.method;

    // ── OPTIONS (CORS preflight) ─────────────────────────────────────
    if (method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

    // ── GET / ────────────────────────────────────────────────────────
    if (method === 'GET' && pathname === '/') {
      try {
        const data = fs.readFileSync(path.join(__dirname, 'index.html'));
        res.setHeader('Content-Type', 'text/html');
        res.writeHead(200);
        res.end(data);
      } catch { res.writeHead(404); res.end(); }
      return;
    }

    // ── GET /search ──────────────────────────────────────────────────
    if (method === 'GET' && pathname === '/search') {
      const params = parseQuery(parsed.search ? parsed.search.slice(1) : '');
      const q      = parseVec(params.v || '');
      if (q.length !== DIMS) { send(res, 400, `{"error":"need ${DIMS}D vector"}`); return; }
      const k      = parseInt(params.k || '5') || 5;
      const metric = params.metric || 'cosine';
      const algo   = params.algo   || 'hnsw';

      const out = db.search(q, k, metric, algo);
      const ss  = JSON.stringify({
        results: out.hits.map(h => ({
          id: h.id, metadata: h.meta, category: h.cat,
          distance: parseFloat(h.dist.toFixed(6)),
          embedding: h.emb
        })),
        latencyUs: out.us, algo: out.algo, metric: out.metric
      });
      send(res, 200, ss);
      return;
    }

    // ── POST /insert ─────────────────────────────────────────────────
    if (method === 'POST' && pathname === '/insert') {
      const body = await readBody(req);
      const pb   = parseBodyFields(body);
      if (!pb.valid || pb.emb.length !== DIMS) { send(res, 400, '{"error":"invalid body"}'); return; }
      const id = db.insert(pb.meta, pb.cat, pb.emb, getDistFn('cosine'));
      send(res, 200, `{"id":${id}}`);
      return;
    }

    // ── DELETE /delete/{id} ──────────────────────────────────────────
    if (method === 'DELETE' && pathname.startsWith('/delete/')) {
      try {
        const id = parseInt(pathname.split('/').pop());
        const ok = db.remove(id);
        send(res, 200, `{"ok":${ok}}`);
      } catch { send(res, 400, '{"error":"invalid id"}'); }
      return;
    }

    // ── GET /items ───────────────────────────────────────────────────
    if (method === 'GET' && pathname === '/items') {
      const items = db.all();
      send(res, 200, JSON.stringify(items.map(v => ({
        id: v.id, metadata: v.metadata, category: v.category, embedding: v.emb
      }))));
      return;
    }

    // ── GET /benchmark ───────────────────────────────────────────────
    if (method === 'GET' && pathname === '/benchmark') {
      const params = parseQuery(parsed.search ? parsed.search.slice(1) : '');
      const q      = parseVec(params.v || '');
      if (q.length !== DIMS) { send(res, 400, `{"error":"need ${DIMS}D vector"}`); return; }
      const k      = parseInt(params.k || '5') || 5;
      const metric = params.metric || 'cosine';
      const b      = db.benchmark(q, k, metric);
      send(res, 200, JSON.stringify({
        bruteforceUs: b.bfUs, kdtreeUs: b.kdUs, hnswUs: b.hnswUs, itemCount: b.n
      }));
      return;
    }

    // ── GET /hnsw-info ───────────────────────────────────────────────
    if (method === 'GET' && pathname === '/hnsw-info') {
      const gi = db.hnswInfo();
      send(res, 200, JSON.stringify({
        topLayer:      gi.topLayer,
        nodeCount:     gi.nodeCount,
        nodesPerLayer: gi.nodesPerLayer,
        edgesPerLayer: gi.edgesPerLayer,
        nodes: gi.nodes.map((n, i) => ({
          id: n[0], metadata: gi.nodesMeta[i], category: gi.nodesCat[i], maxLyr: n[1]
        })),
        edges: gi.edges.map(e => ({ src: e[0], dst: e[1], lyr: e[2] }))
      }));
      return;
    }

    // ── POST /doc/insert ─────────────────────────────────────────────
    if (method === 'POST' && pathname === '/doc/insert') {
      const body  = await readBody(req);
      let parsed2;
      try { parsed2 = JSON.parse(body); } catch { send(res, 400, '{"error":"invalid JSON"}'); return; }
      const title = parsed2.title || '';
      const text  = parsed2.text  || '';
      if (!title || !text) { send(res, 400, '{"error":"need title and text"}'); return; }

      const chunks = chunkText(text, 250, 30);
      const ids    = [];
      for (let i = 0; i < chunks.length; i++) {
        const emb = await ollama.embed(chunks[i]);
        if (!emb.length) {
          send(res, 503, JSON.stringify({ error: 'Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2' }));
          return;
        }
        const chunkTitle = chunks.length > 1 ? `${title} [${i+1}/${chunks.length}]` : title;
        ids.push(docDB.insert(chunkTitle, chunks[i], emb));
      }
      send(res, 200, JSON.stringify({ ids, chunks: chunks.length, dims: docDB.dims }));
      return;
    }

    // ── DELETE /doc/delete/{id} ──────────────────────────────────────
    if (method === 'DELETE' && pathname.startsWith('/doc/delete/')) {
      try {
        const id = parseInt(pathname.split('/').pop());
        const ok = docDB.remove(id);
        send(res, 200, `{"ok":${ok}}`);
      } catch { send(res, 400, '{"error":"invalid id"}'); }
      return;
    }

    // ── GET /doc/list ────────────────────────────────────────────────
    if (method === 'GET' && pathname === '/doc/list') {
      const docs = docDB.all();
      send(res, 200, JSON.stringify(docs.map(d => {
        const preview = d.text.length > 120 ? d.text.slice(0, 120) + '…' : d.text;
        const words   = d.text.split(/\s+/).length;
        return { id: d.id, title: d.title, preview, words };
      })));
      return;
    }

    // ── POST /doc/search ─────────────────────────────────────────────
    if (method === 'POST' && pathname === '/doc/search') {
      const body = await readBody(req);
      let obj;
      try { obj = JSON.parse(body); } catch { send(res, 400, '{"error":"invalid JSON"}'); return; }
      const question = obj.question || '';
      const k        = obj.k || 3;
      if (!question) { send(res, 400, '{"error":"need question"}'); return; }

      const qEmb = await ollama.embed(question);
      if (!qEmb.length) { send(res, 503, '{"error":"Ollama unavailable"}'); return; }

      const hits = docDB.search(qEmb, k, 0.7);
      const contexts = [];
      for (const r of hits) {
        const id   = r[1];
        const dist = r[0];
        const d    = docDB.all().find(di => di.id === id);
        if (d) contexts.push({ id, title: d.title, distance: parseFloat(dist.toFixed(4)) });
      }
      send(res, 200, JSON.stringify({ contexts }));
      return;
    }

    // ── POST /doc/ask ────────────────────────────────────────────────
    if (method === 'POST' && pathname === '/doc/ask') {
      const body = await readBody(req);
      let obj;
      try { obj = JSON.parse(body); } catch { send(res, 400, '{"error":"invalid JSON"}'); return; }
      const question = obj.question || '';
      const k        = obj.k || 3;
      if (!question) { send(res, 400, '{"error":"need question"}'); return; }

      // Step 1: embed
      const qEmb = await ollama.embed(question);
      if (!qEmb.length) { send(res, 503, '{"error":"Ollama unavailable"}'); return; }

      // Step 2: retrieve
      const rawHits = docDB.search(qEmb, k, 0.7);
      const hitDocs = [];
      const hitDist = [];
      const allDocs = docDB.all();
      for (const r of rawHits) {
        const d = allDocs.find(di => di.id === r[1]);
        if (d) { hitDocs.push(d); hitDist.push(r[0]); }
      }

      // Step 3: build prompt
      let ctx = '';
      for (let i = 0; i < hitDocs.length; i++)
        ctx += `[${i+1}] ${hitDocs[i].title}:\n${hitDocs[i].text}\n\n`;

      const prompt =
        'You are a helpful assistant. Answer the user\'s question directly. '
        + 'Use the provided context if it contains relevant information. '
        + 'If it doesn\'t, just use your own general knowledge. '
        + 'IMPORTANT: Do NOT mention the \'context\', \'provided text\', or say things like '
        + '\'the context doesn\'t mention\'. Just answer the question naturally.\n\n'
        + `Context:\n${ctx}`
        + `Question: ${question}\n\nAnswer:`;

      // Step 4: generate
      const answer = await ollama.generate(prompt);

      // Step 5: return
      send(res, 200, JSON.stringify({
        answer,
        model:    ollama.genModel,
        contexts: hitDocs.map((d, i) => ({
          id: d.id, title: d.title, text: d.text,
          distance: parseFloat(hitDist[i].toFixed(4))
        })),
        docCount: docDB.size
      }));
      return;
    }

    // ── GET /status ──────────────────────────────────────────────────
    if (method === 'GET' && pathname === '/status') {
      const up = await ollama.isAvailable();
      send(res, 200, JSON.stringify({
        ollamaAvailable: up,
        embedModel:  ollama.embedModel,
        genModel:    ollama.genModel,
        docCount:    docDB.size,
        docDims:     docDB.dims,
        demoDims:    DIMS,
        demoCount:   db.size
      }));
      return;
    }

    // ── GET /stats ───────────────────────────────────────────────────
    if (method === 'GET' && pathname === '/stats') {
      send(res, 200, JSON.stringify({
        count:      db.size,
        dims:       DIMS,
        algorithms: ['bruteforce', 'kdtree', 'hnsw'],
        metrics:    ['euclidean', 'cosine', 'manhattan']
      }));
      return;
    }

    // ── 404 ──────────────────────────────────────────────────────────
    send(res, 404, '{"error":"not found"}');
  });

  server.listen(8080, () => {
    console.log('Server started on http://localhost:8080');
  });
}

main().catch(err => { console.error(err); process.exit(1); });