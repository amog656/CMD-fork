/**
 * GANDIVA Backend Server
 * Connects the frontend GANDIVA UI to the backend generation pipeline and model APIs.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const PORT = process.env.PORT || 3000;

const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.pdb': 'text/plain',
  '.sdf': 'text/plain',
  '.txt': 'text/plain'
};

const server = http.createServer((req, res) => {
  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname;

  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // Health check endpoint
  if (pathname === '/api/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', server: 'GANDIVA Backend' }));
    return;
  }

  // Pipeline generation endpoint: connects frontend window.apo2mol.generate to your model
  if (pathname === '/api/generate' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const payload = JSON.parse(body || '{}');
        const pdbText = payload.pdb || '';
        const fileName = payload.name || '';
        const options = payload.options || {};

        // If you have a local Python model running (e.g. port 8000 / CMD-GEN), forward the call:
        const pythonEndpoint = process.env.MODEL_URL;
        if (pythonEndpoint) {
          const reqPy = http.request(pythonEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
          }, pyRes => {
            res.writeHead(pyRes.statusCode, { 'Content-Type': 'text/plain' });
            pyRes.pipe(res);
          });
          reqPy.on('error', () => {
            // If external model unavailable, return 204 to let frontend use native generation
            res.writeHead(204);
            res.end();
          });
          reqPy.write(JSON.stringify({ pdb: pdbText, name: fileName, count: options.count, site: options.site }));
          reqPy.end();
          return;
        }

        // Default response: return 204 (No Content) so the frontend uses its native procedural engine
        res.writeHead(204);
        res.end();
      } catch (err) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // Static file server
  let filePath = path.join(__dirname, pathname === '/' ? 'GANDIVA.html' : pathname);
  if (!filePath.startsWith(__dirname)) {
    res.writeHead(403);
    res.end('Access Denied');
    return;
  }

  fs.stat(filePath, (err, stats) => {
    if (err || !stats.isFile()) {
      filePath = path.join(__dirname, 'GANDIVA.html');
    }
    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'text/html';
    res.writeHead(200, { 'Content-Type': contentType });
    fs.createReadStream(filePath).pipe(res);
  });
});

server.listen(PORT, () => {
  console.log(`GANDIVA server running at: http://localhost:${PORT}`);
});
