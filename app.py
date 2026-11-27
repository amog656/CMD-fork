from __future__ import annotations

import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Sequence

import streamlit as st
from complex_input import extract_complex

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PHARMFLOW_CKPT = PROJECT_ROOT / "checkpoints_pharmflow" / "best_model_epoch_46.ckpt"
DEFAULT_GCPG_PYTHON = Path(r"C:\Users\hanut sai\miniconda3\envs\gcpg\python.exe")
DEFAULT_MEEKO_PYTHON = Path(r"C:\Users\hanut sai\miniconda3\envs\diff_phar\python.exe")
DEFAULT_VINA = PROJECT_ROOT / "vina.exe"
def _resolved_file(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _pipeline_command(
    pdb_path: Path,
    out_dir: Path,
    center: Sequence[float],
    radius: float,
    pocket_note: str,
    checkpoint: Path,
    gcpg_python: Path,
    meeko_python: Path,
    vina: Path,
    states: int,
    molecules: int,
    exhaustiveness: int,
    device: str,
    allow_bad_residues: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "run_full_pipeline.py"),
        "--pdb", str(pdb_path),
        "--pocket_center", *(f"{value:g}" for value in center),
        "--pocket_radius", f"{radius:g}",
        "--pocket_note", pocket_note,
        "--k_states", str(states),
        "--ckpt", str(checkpoint),
        "--gcpg_python", str(gcpg_python),
        "--meeko_python", str(meeko_python),
        "--vina_exe", str(vina),
        "--n_mol", str(molecules),
        "--exhaustiveness", str(exhaustiveness),
        "--device", device,
        "--out_dir", str(out_dir),
    ]
    if allow_bad_residues:
        command.append("--allow_bad_receptor_residues")
    return command


def _run_pipeline(command: list[str], output_dir: Path) -> tuple[int, str]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    log_lines: list[str] = []
    log_view = st.empty()
    assert process.stdout is not None
    for line in process.stdout:
        log_lines.append(line.rstrip())
        log_view.code("\n".join(log_lines[-70:]), language="text")
    return process.wait(), "\n".join(log_lines)


st.set_page_config(page_title="DiffPhar", page_icon="🧬", layout="wide")
st.title("DiffPhar molecule generator")
st.caption("Upload one complete protein-ligand complex. The app extracts the protein and finds the pocket center from its bound ligand.")

st.warning(
    "Generated molecules and Vina scores are computational predictions. "
    "They do not establish that a molecule is an effective or safe drug."
)

st.subheader("1. Protein and pocket")
with st.form("pipeline_form"):
    complex_upload = st.file_uploader("Protein-ligand complex (.pdb, .cif, or .mmcif)", type=["pdb", "ent", "cif", "mmcif"])
    pocket_radius = 10.0
    st.info("Upload a complete complex with a small molecule bound. The ligand is used to locate the pocket automatically; you do not enter coordinates.")

    with st.expander("Advanced settings (optional)"):
        st.caption("Defaults are prefilled. Change these only if you need a different run size or installed model path.")
        config_cols = st.columns(3)
        with config_cols[0]:
            checkpoint_text = st.text_input("PharmFlow checkpoint", value=str(DEFAULT_PHARMFLOW_CKPT))
            states = st.slider("Protein states", min_value=1, max_value=12, value=2)
        with config_cols[1]:
            gcpg_python_text = st.text_input("GCPG Python", value=str(DEFAULT_GCPG_PYTHON))
            molecules = st.number_input("Molecules to generate", min_value=1, max_value=100, value=6, step=1)
            exhaustiveness = st.slider("Vina exhaustiveness", min_value=1, max_value=16, value=1,
                                       help="Lower is faster. 1 is fast; use 8 for more thorough docking.")
        with config_cols[2]:
            meeko_python_text = st.text_input("Meeko Python", value=str(DEFAULT_MEEKO_PYTHON))
            st.caption("Use the diff_phar environment; its RDKit is compatible with Meeko 0.8.")
            device = st.selectbox(
                "Compute device (PharmFlow + aSAM)",
                ["cuda", "cpu"],
                index=0,
                help="cuda uses the GPU for PharmFlow and aSAM; aSAM falls back to CPU if CUDA fails. GCPG and Vina are CPU-only.",
            )
        vina_text = st.text_input("AutoDock Vina executable", value=str(DEFAULT_VINA))
        allow_bad_residues = st.checkbox(
            "Allow Meeko to omit unmatched receptor residues",
            value=True,
            help="Prepared examples may include residues without a Meeko template. Omitted residues can affect docking; inspect the run output before interpreting scores.",
        )
        st.caption("Default quick run: 2 protein states, 6 molecules, Vina exhaustiveness 1.")

    submitted = st.form_submit_button("Generate and dock molecules", type="primary", use_container_width=True)

if submitted:
    if complex_upload is None:
        st.error("Upload a complete protein-ligand complex first.")
    else:
        try:
            uploaded_receptor, center, ligand_id = extract_complex(complex_upload.getvalue(), complex_upload.name)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        checkpoint = _resolved_file(checkpoint_text)
        gcpg_python = _resolved_file(gcpg_python_text)
        meeko_python = _resolved_file(meeko_python_text)
        vina = _resolved_file(vina_text)
        gcpg_model = PROJECT_ROOT / "checkpoints_gcpg/model_checkpoints/checkpoints/final_model.pth"
        gcpg_tokenizer = PROJECT_ROOT / "checkpoints_gcpg/model_checkpoints/checkpoints/tokenizer_r_iso.pkl"
        missing = [str(path) for path in [checkpoint, gcpg_python, meeko_python, vina, gcpg_model, gcpg_tokenizer] if not path.is_file()]
        if missing:
            st.error("These required files are missing:\n\n" + "\n".join(f"- `{item}`" for item in missing))
        else:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            output_dir = PROJECT_ROOT / "pipeline_runs" / run_name
            input_dir = output_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=False)
            pdb_path = input_dir / "receptor.pdb"
            original_name = Path(complex_upload.name).name
            (input_dir / original_name).write_bytes(complex_upload.getvalue())
            pdb_path.write_bytes(uploaded_receptor)
            pocket_note = f"Automatically detected bound ligand ({ligand_id}) in uploaded complex {original_name}; heavy-atom centroid"
            st.success(f"Found {ligand_id}. Pocket center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) Å; radius {pocket_radius:.1f} Å.")
            command = _pipeline_command(
                pdb_path=pdb_path,
                out_dir=output_dir,
                center=center,
                radius=pocket_radius,
                pocket_note=pocket_note,
                checkpoint=checkpoint,
                gcpg_python=gcpg_python,
                meeko_python=meeko_python,
                vina=vina,
                states=states,
                molecules=int(molecules),
                exhaustiveness=int(exhaustiveness),
                device=device,
                allow_bad_residues=allow_bad_residues,
            )
            st.session_state["last_run_dir"] = str(output_dir)
            st.session_state["last_run_exit"] = None
            st.subheader("Pipeline progress")
            st.code(subprocess.list2cmdline(command), language="powershell")
            with st.status("Running the five model stages…", expanded=True) as status:
                try:
                    exit_code, run_log = _run_pipeline(command, output_dir)
                    (output_dir / "frontend_run.log").write_text(run_log, encoding="utf-8")
                    st.session_state["last_run_exit"] = exit_code
                    if exit_code == 0:
                        status.update(label="Pipeline finished", state="complete", expanded=False)
                    else:
                        status.update(label=f"Pipeline stopped (exit code {exit_code})", state="error", expanded=True)
                except Exception as exc:
                    status.update(label="Could not start the pipeline", state="error", expanded=True)
                    st.exception(exc)

last_run = st.session_state.get("last_run_dir")
last_exit = st.session_state.get("last_run_exit")
if last_run:
    run_dir = Path(last_run)
    report = run_dir / "PIPELINE_VALIDATION_REPORT.md"
    sdf = run_dir / "step4_molecules/consensus_blueprint_candidates_3d.sdf"
    smiles = run_dir / "step4_molecules/consensus_blueprint_generated_smiles.txt"
    st.subheader("Latest run")
    st.code(str(run_dir), language="text")
    if last_exit == 0 and report.is_file():
        if sdf.is_file():
            st.download_button("Download 3D molecules (SDF)", sdf.read_bytes(), file_name=sdf.name, mime="chemical/x-mdl-sdfile")
        if smiles.is_file():
            st.download_button("Download SMILES", smiles.read_bytes(), file_name=smiles.name, mime="text/plain")
        with st.expander("View run report", expanded=True):
            st.markdown(report.read_text(encoding="utf-8"))
    elif last_exit not in (None, 0):
        st.error("The run did not finish. Check `frontend_run.log` in the run folder for the full error output.")
