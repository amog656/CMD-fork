"""GANDIVA Web Bridge: Connects the GANDIVA Web UI to the PharmFlow + GCPG + Vina pipeline."""

from __future__ import annotations

import io
import json
import mimetypes
import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, QED

from .adapters.cmd_flow_matching import CMDFlowMatchingAdapter
from .adapters.cmd_gcpg import CMDGCPGAdapter
from .adapters.docking import VinaDockingAdapter
from .data.models import PharmModel, PharmPoint, PocketAtom, PocketState

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

FRONTEND_DIR = Path(__file__).parents[1] / "frontend" / "INHIBITOR SITE"


def generate_molecules_sdf(
    pdb_text: str,
    name: str = "target.pdb",
    count: int = 4,
    site: list[float] | None = None,
) -> str:
    """Run full PharmFlow -> GCPG -> Vina docking and format into standard SDF for GANDIVA 3D viewer."""
    with tempfile.TemporaryDirectory(prefix="gandiva-run-") as td:
        clean_name = Path(name).name if name else "target.pdb"
        if not clean_name.endswith(".pdb"):
            clean_name += ".pdb"
        pdb_path = Path(td) / clean_name
        pdb_path.write_text(pdb_text, encoding="utf-8")

        pocket_atoms = []
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("ATOM"):
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        element = line[76:78].strip() or line[12:16].strip()[0]
                        res_name = line[17:20].strip()
                        res_id = line[22:26].strip()
                        atom_name = line[12:16].strip()
                        pocket_atoms.append(PocketAtom((x, y, z), element, res_name, res_id, atom_name))
                    except Exception:
                        continue

        if not pocket_atoms:
            raise ValueError("No ATOM records found in the provided PDB data.")

        # Determine pocket center (from site, ligand HETATMs, or centroid)
        pocket_center = site
        if not pocket_center:
            het_coords = []
            with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.startswith("HETATM") and "HOH" not in line:
                        try:
                            hx = float(line[30:38].strip())
                            hy = float(line[38:46].strip())
                            hz = float(line[46:54].strip())
                            het_coords.append((hx, hy, hz))
                        except Exception:
                            continue
            if het_coords:
                pocket_center = [
                    float(sum(c[0] for c in het_coords) / len(het_coords)),
                    float(sum(c[1] for c in het_coords) / len(het_coords)),
                    float(sum(c[2] for c in het_coords) / len(het_coords)),
                ]

        # For large whole-protein files, isolate the active pocket within 14 Angstroms
        if len(pocket_atoms) > 400 and pocket_center:
            px, py, pz = pocket_center
            near_atoms = [
                a for a in pocket_atoms
                if ((a.xyz[0] - px)**2 + (a.xyz[1] - py)**2 + (a.xyz[2] - pz)**2)**0.5 <= 14.0
            ]
            if len(near_atoms) >= 20:
                pocket_atoms = near_atoms

        state = PocketState(
            state_id=pdb_path.stem,
            source="crystal",
            coordinate_path=pdb_path,
            pocket_atoms=pocket_atoms,
            in_domain=True,
        )

        # Check for co-crystallized ligand in pocket
        het_lines = []
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("HETATM") and "HOH" not in line and "MSE" not in line:
                    het_lines.append(line)

        model = None
        if het_lines:
            try:
                from rdkit import RDConfig
                from rdkit.Chem import ChemicalFeatures
                import numpy as np

                lig_mol = Chem.MolFromPDBBlock("".join(het_lines))
                if lig_mol and lig_mol.GetNumAtoms() > 3:
                    fdef = str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
                    feat_factory = ChemicalFeatures.BuildFeatureFactory(fdef)
                    feats = feat_factory.GetFeaturesForMol(lig_mol)
                    conf = lig_mol.GetConformer()

                    mapping = {
                        "Donor": "HDON",
                        "Acceptor": "HACC",
                        "Hydrophobe": "HYBL",
                        "Aromatic": "AROM",
                        "PosIonizable": "POSC",
                        "NegIonizable": "NEGI",
                    }
                    extracted_points = []
                    for feat in feats:
                        fam = feat.GetFamily()
                        ft = mapping.get(fam)
                        if ft:
                            pos = np.mean([list(conf.GetAtomPosition(a)) for a in feat.GetAtomIds()], axis=0)
                            extracted_points.append(PharmPoint(tuple(np.round(pos, 3)), ft))

                    if extracted_points:
                        sel_points = extracted_points[:8]
                        model = PharmModel(
                            model_id=f"crystal_{pdb_path.stem}",
                            points=sel_points,
                            origin="per-state",
                        )
                        print(f"[GANDIVA] Extracted {len(sel_points)} ground-truth crystal pharmacophore points from pocket ligand.")
            except Exception as e:
                print(f"[GANDIVA] Crystal pharmacophore extraction skipped: {e}")

        # Fallback to PharmFlow if no ligand or extraction failed
        if model is None:
            print("[GANDIVA] Running PharmFlow Flow-Matching AI pharmacophore prediction...")
            flow_adapter = CMDFlowMatchingAdapter(
                repository_dir="pharmflow_repo",
                checkpoint_path="pharmflow_repo/best-model-epoch=epoch=01.ckpt",
                samples_per_state=1,
                timesteps=10,
            )
            model = flow_adapter.predict(state, seed=42)

        # 2. GCPG Molecule generation
        gcpg_adapter = CMDGCPGAdapter(
            repository_dir="cmdgen_repo",
            model_path="cmdgen_repo/GCPG/checkpoints/model_checkpoints/checkpoints/final_model.pth",
            tokenizer_path="cmdgen_repo/GCPG/checkpoints/model_checkpoints/checkpoints/tokenizer_r_iso.pkl",
            device="cpu",
        )
        smiles_list = gcpg_adapter.generate(model, count=count, property_gates={}, seed=42)

        # 3. AutoDock Vina physical docking
        vina_adapter = VinaDockingAdapter(
            vina_executable="bin/vina.exe",
            receptor_paths={pdb_path.stem: pdb_path},
            exhaustiveness=2,
        )

        sdf_blocks = []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                continue
            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, randomSeed=42 + i) < 0:
                AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42 + i)
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass

            score = vina_adapter.score(smi, pdb_path.stem, seed=42 + i, box_center=pocket_center)
            qed_val = float(QED.qed(mol))

            mol.SetProp("_Name", f"Candidate {i+1}")
            mol.SetProp("affinity", f"{score:.2f}")
            mol.SetProp("score", f"{score:.2f}")
            mol.SetProp("qed", f"{qed_val:.2f}")
            mol.SetProp("smiles", smi)

            sio = io.StringIO()
            writer = Chem.SDWriter(sio)
            writer.write(mol)
            writer.close()
            sdf_blocks.append(sio.getvalue())

        return "".join(sdf_blocks)


class GandivaHandler(BaseHTTPRequestHandler):
    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "backend": "GANDIVA Python Pipeline"}).encode("utf-8"))
            return

        # Serve static files from FRONTEND_DIR
        clean_path = self.path.split("?")[0].lstrip("/")
        if not clean_path:
            clean_path = "GANDIVA.html"

        file_path = (FRONTEND_DIR / clean_path).resolve()
        if not str(file_path).startswith(str(FRONTEND_DIR.resolve())) or not file_path.is_file():
            # Fallback to GANDIVA.html
            file_path = FRONTEND_DIR / "GANDIVA.html"

        if file_path.is_file():
            mime, _ = mimetypes.guess_type(str(file_path))
            mime = mime or "text/html"
            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", mime)
            self.end_headers()
            self.wfile.write(file_path.read_bytes())
        else:
            self.send_response(404)
            self._send_cors()
            self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode("utf-8"))
                pdb_text = payload.get("pdb", "")
                name = payload.get("name", "uploaded.pdb")
                options = payload.get("options", {}) or {}
                count = int(options.get("count") or payload.get("count") or 4)
                site = options.get("site") or payload.get("site")

                if not pdb_text.strip():
                    self.send_response(400)
                    self._send_cors()
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Missing pdb content"}).encode("utf-8"))
                    return

                print(f"[GANDIVA] Generating {count} molecules for '{name}' via PharmFlow + GCPG + Vina...")
                sdf_output = generate_molecules_sdf(pdb_text, name=name, count=count, site=site)
                print(f"[GANDIVA] Generation complete. Sending {len(sdf_output)} bytes of 3D SDF to UI.")

                self.send_response(200)
                self._send_cors()
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(sdf_output.encode("utf-8"))
            except Exception as e:
                print(f"[GANDIVA Error] {e}")
                self.send_response(500)
                self._send_cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self._send_cors()
            self.end_headers()


def start_server(port: int = 3000) -> None:
    server = HTTPServer(("0.0.0.0", port), GandivaHandler)
    print(f"================================================================")
    print(f"  GANDIVA Web UI & Full Pipeline Server Active")
    print(f"  URL: http://localhost:{port}")
    print(f"================================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down GANDIVA server.")
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    start_server(port)
