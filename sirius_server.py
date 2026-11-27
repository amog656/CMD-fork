"""Local web API and static server for the SIRIUS CMD-Gen frontend."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from email.parser import BytesParser
from email.policy import default
import subprocess
import sys
import urllib.parse

from complex_input import extract_complex

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "pipeline_runs"
RUNS.mkdir(exist_ok=True)
RUNS_STATE: dict[str, dict] = {}
LOCK = threading.Lock()


def _parse_multipart(content_type: str, body: bytes) -> dict[str, tuple[str, bytes]]:
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: " + content_type.encode("ascii") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    fields = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        fields[name] = (filename or "", payload)
    return fields


def _candidate_results(out_dir: Path) -> list[dict]:
    report = out_dir / "PIPELINE_VALIDATION_REPORT.md"
    sdf = out_dir / "step4_molecules" / "consensus_blueprint_candidates_3d.sdf"
    if not report.is_file() or not sdf.is_file():
        return []
    scores = {}
    row_re = re.compile(r"\| \*\*#(\d+)\*\* \| `([^`]+)` \| `([^`]+)` \| \*\*(-?[\d.]+)\*\* \| (-?[\d.]+) \| ([\d.]+) \|")
    for line in report.read_text(encoding="utf-8").splitlines():
        match = row_re.search(line)
        if match:
            rank, name, smiles, mean, worst, variance = match.groups()
            scores[name] = {
                "rank": int(rank), "smiles": smiles,
                "dk": float(mean), "worst": float(worst), "variance": float(variance),
            }
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, QED, rdDepictor
    except ImportError:
        return []
    results = []
    for mol in Chem.SDMolSupplier(str(sdf), removeHs=True):
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else "candidate"
        if name not in scores:
            continue
        score = scores[name]
        mol2d = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(mol2d)
        conf = mol2d.GetConformer()
        atoms = [{"x": float(conf.GetAtomPosition(i).x), "y": float(conf.GetAtomPosition(i).y),
                  "z": 0.0, "e": atom.GetSymbol()} for i, atom in enumerate(mol2d.GetAtoms())]
        # Preserve the original SDF conformer for the Structure Viewer. These
        # are generated, pre-docking coordinates in Angstroms, not Vina poses.
        atoms3d = None
        if mol.GetNumConformers():
            conf3d = mol.GetConformer()
            atoms3d = [{"x": float(conf3d.GetAtomPosition(i).x),
                        "y": float(conf3d.GetAtomPosition(i).y),
                        "z": float(conf3d.GetAtomPosition(i).z),
                        "e": atom.GetSymbol()} for i, atom in enumerate(mol.GetAtoms())]
        bonds = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol2d.GetBonds()]
        results.append({
            "n": name, "rank": score["rank"], "smiles": score["smiles"],
            "mw": float(Descriptors.MolWt(mol)), "qed": float(QED.qed(mol)),
            "sas": None, "lp": float(Crippen.MolLogP(mol)), "dk": score["dk"],
            "worst": score["worst"], "variance": score["variance"],
            "d": score["smiles"], "m": {"at": atoms, "bd": bonds, "subs": []},
            "m3d": {"at": atoms3d, "bd": bonds, "subs": []} if atoms3d else None,
            "m3d_source": "Generated RDKit/UFF conformer from candidate SDF; pre-docking" if atoms3d else None,
        })
    return sorted(results, key=lambda item: item["rank"])


def _run_job(run_id: str, command: list[str], out_dir: Path):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    try:
        with LOCK:
            RUNS_STATE[run_id]["status"] = "running"
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1)
        with LOCK:
            RUNS_STATE[run_id]["pid"] = proc.pid
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            with LOCK:
                state = RUNS_STATE[run_id]
                state["log"].append(line)
                state["log"] = state["log"][-150:]
                if line.startswith(">>> STEP"):
                    state["phase"] = line.split(":", 1)[-1].strip()
                    state["progress"] = min(90, state["progress"] + 18)
        code = proc.wait()
        if code == 0:
            candidates = _candidate_results(out_dir)
            with LOCK:
                state = RUNS_STATE[run_id]
                state.update(status="complete", progress=100, phase="Complete", candidates=candidates,
                             report_url=f"/api/runs/{run_id}/report", sdf_url=f"/api/runs/{run_id}/sdf")
                if not candidates:
                    state.update(status="error", error="Pipeline finished but candidate results could not be read.")
        else:
            with LOCK:
                RUNS_STATE[run_id].update(status="error", error=f"Pipeline exited with code {code}.")
    except Exception as exc:
        with LOCK:
            RUNS_STATE[run_id].update(status="error", error=str(exc))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _json(self, value: dict, status: int = 200):
        data = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/health":
            return self._json({"ok": True, "service": "SIRIUS CMD-Gen"})
        match = re.fullmatch(r"/api/runs/([a-f0-9]+)/?(report|sdf)?", path)
        if match:
            run_id, artifact = match.groups()
            with LOCK:
                state = RUNS_STATE.get(run_id)
                snapshot = dict(state) if state else None
            # Completed run outputs live on disk, so recover them after a web
            # server restart. This also lets the viewer load newly exposed 3D
            # conformer coordinates for an already completed run.
            if not snapshot and re.fullmatch(r"[a-f0-9]{32}", run_id):
                matches = list(RUNS.glob(f"*_sirius_{run_id[:8]}"))
                if len(matches) == 1:
                    out_dir = matches[0]
                    candidates = _candidate_results(out_dir)
                    if candidates:
                        snapshot = {"id": run_id, "status": "complete", "progress": 100,
                                    "phase": "Complete", "stage": 5, "log": [],
                                    "candidates": candidates, "out_dir": str(out_dir),
                                    "report_url": f"/api/runs/{run_id}/report",
                                    "sdf_url": f"/api/runs/{run_id}/sdf"}
            if not snapshot:
                return self._json({"error": "Run not found or its completed candidate files are unavailable."}, 404)
            if artifact:
                out_dir = Path(snapshot["out_dir"])
                file_path = out_dir / ("PIPELINE_VALIDATION_REPORT.md" if artifact == "report" else
                                       "step4_molecules/consensus_blueprint_candidates_3d.sdf")
                if not file_path.is_file():
                    return self._json({"error": "Requested run file is not available."}, 404)
                content_type = "text/markdown; charset=utf-8" if artifact == "report" else "chemical/x-mdl-sdfile"
                data = file_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Disposition", f'attachment; filename="{file_path.name}"')
                self.end_headers()
                self.wfile.write(data)
                return
            phase_index = max(0, next((i for i, label in enumerate(("aSAM", "PharmFlow", "Consensus", "GCPG", "Docking"))
                                       if label.lower() in snapshot.get("phase", "").lower()), 0))
            snapshot["stage"] = phase_index
            snapshot.pop("out_dir", None)
            snapshot.pop("pid", None)
            return self._json(snapshot)
        if path in {"/", "/index.html", "/app.js", "/style.css"}:
            return super().do_GET()
        return self._json({"error": "Not found."}, 404)

    def do_POST(self):
        origin = self.headers.get("Origin")
        allowed_origin = "http://" + self.headers.get("Host", "127.0.0.1:8000")
        if origin and origin != allowed_origin:
            return self._json({"error": "Cross-origin requests are not allowed."}, 403)
        if urllib.parse.urlparse(self.path).path != "/api/runs":
            return self._json({"error": "Not found."}, 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100 * 1024 * 1024:
                return self._json({"error": "Upload must be between 1 byte and 100 MB."}, 400)
            fields = _parse_multipart(self.headers.get("Content-Type", ""), self.rfile.read(length))
            filename, complex_bytes = fields.get("complex", ("", b""))
            if not filename or not complex_bytes:
                return self._json({"error": "Upload a complete protein-ligand complex."}, 400)
            receptor, center, ligand_id = extract_complex(complex_bytes, Path(filename).name)
            states = max(1, min(12, int(fields.get("states", ("", b"2"))[1].decode() or 2)))
            molecules = max(1, min(100, int(fields.get("molecules", ("", b"6"))[1].decode() or 6)))
            exhaustiveness = max(1, min(16, int(fields.get("exhaustiveness", ("", b"1"))[1].decode() or 1)))
            # Keep the web app on its GPU-enabled inference path.
            device = "cuda"
        except (ValueError, UnicodeDecodeError) as exc:
            return self._json({"error": str(exc)}, 400)

        run_id = uuid.uuid4().hex
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_sirius_" + run_id[:8]
        out_dir = RUNS / run_name
        input_dir = out_dir / "input"
        input_dir.mkdir(parents=True)
        (input_dir / Path(filename).name).write_bytes(complex_bytes)
        receptor_path = input_dir / "receptor.pdb"
        receptor_path.write_bytes(receptor)
        checkpoint = ROOT / "checkpoints_pharmflow" / "best_model_epoch_46.ckpt"
        gcpg_python = Path(r"C:\Users\hanut sai\miniconda3\envs\gcpg\python.exe")
        meeko_python = Path(r"C:\Users\hanut sai\miniconda3\envs\diff_phar\python.exe")
        vina = ROOT / "vina.exe"
        required = [checkpoint, gcpg_python, meeko_python, vina,
                    ROOT / "checkpoints_gcpg/model_checkpoints/checkpoints/final_model.pth",
                    ROOT / "checkpoints_gcpg/model_checkpoints/checkpoints/tokenizer_r_iso.pkl"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            return self._json({"error": "Missing required model/runtime files: " + ", ".join(missing)}, 500)
        pocket_note = f"Detected bound ligand {ligand_id} in {Path(filename).name}; heavy-atom centroid"
        command = [sys.executable, str(ROOT / "run_full_pipeline.py"), "--pdb", str(receptor_path),
                   "--pocket_center", *(f"{v:.6f}" for v in center), "--pocket_radius", "10",
                   "--pocket_note", pocket_note, "--k_states", str(states), "--ckpt", str(checkpoint),
                   "--gcpg_python", str(gcpg_python), "--meeko_python", str(meeko_python),
                   "--vina_exe", str(vina), "--n_mol", str(molecules), "--exhaustiveness", str(exhaustiveness),
                   "--device", device, "--out_dir", str(out_dir), "--allow_bad_receptor_residues"]
        with LOCK:
            RUNS_STATE[run_id] = {"id": run_id, "status": "queued", "progress": 0,
                                  "phase": "Preparing run", "stage": 0, "log": [],
                                  "ligand": ligand_id, "center": center, "out_dir": str(out_dir),
                                  "candidates": []}
        thread = threading.Thread(target=_run_job, args=(run_id, command, out_dir), daemon=True)
        thread.start()
        return self._json({"id": run_id, "status": "queued", "ligand": ligand_id, "center": center})

    def log_message(self, fmt, *args):
        # Keep the terminal quiet; pipeline logs are available from the run API.
        pass


if __name__ == "__main__":
    port = int(os.environ.get("SIRIUS_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"SIRIUS frontend and CMD-Gen API running at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping SIRIUS server.")
        server.server_close()
