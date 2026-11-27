from __future__ import annotations

import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Sequence

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PHARMFLOW_CKPT = PROJECT_ROOT / "checkpoints_pharmflow" / "best_model_epoch_46.ckpt"
DEFAULT_GCPG_PYTHON = Path(r"C:\Users\hanut sai\miniconda3\envs\gcpg\python.exe")
DEFAULT_MEEKO_PYTHON = Path(r"C:\Users\hanut sai\miniconda3\envs\diff_phar\python.exe")
DEFAULT_VINA = PROJECT_ROOT / "vina.exe"
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def _resolved_file(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _protein_receptor_bytes(pdb_bytes: bytes, filename: str) -> bytes:
    """Keep one standard amino-acid chain and remove ligands/water from a PDB upload."""
    lines = pdb_bytes.decode("utf-8", errors="replace").splitlines()
    chains = {
        line[21:22] or " "
        for line in lines
        if line.startswith("ATOM  ") and line[17:20].strip() in STANDARD_AA
    }
    if not chains:
        raise ValueError(
            f"`{filename}` has no standard protein residues (ATOM records). It looks like a ligand-only file. "
            "Choose a built-in protein example, or upload the receptor protein PDB; files ending in "
            "`_bound_reference.pdb` contain only the ligand."
        )
    if "A" in chains:
        chain = "A"
    elif len(chains) == 1:
        chain = next(iter(chains))
    else:
        raise ValueError(
            "This PDB has multiple protein chains and no chain A. aSAM expects a single chain; "
            "upload a receptor PDB with the intended chain labeled A."
        )

    protein_lines = []
    for line in lines:
        if not line.startswith("ATOM  ") or line[17:20].strip() not in STANDARD_AA:
            continue
        if (line[21:22] or " ") != chain:
            continue
        if line[76:78].strip().upper() in {"H", "D"}:
            continue
        protein_lines.append(line[:21] + "A" + line[22:])
    if not protein_lines:
        raise ValueError("No protein atoms remained after filtering the uploaded PDB.")
    return ("\n".join([*protein_lines, "TER", "END", ""])).encode("utf-8")


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
st.caption("Choose a protein pocket, generate candidate molecules, and review docking results.")
st.caption("For custom input, upload a protein receptor PDB. Bound ligand reference files are not receptor structures.")

st.warning(
    "Generated molecules and Vina scores are computational predictions. "
    "They do not establish that a molecule is an effective or safe drug."
)

SAMPLE_EXAMPLES = {
    "Custom protein (upload PDB)": None,
    "FKBP12 · rapamycin (1FKB)": {
        "pdb": "sample_examples/receptors/1FKB_receptor.pdb",
        "center": (3.109, 12.167, 13.353),
        "ligand": "Rapamycin (RAP)",
        "source": "RCSB PDB 1FKB; center is the bound ligand heavy-atom centroid",
    },
    "EGFR kinase · erlotinib (1M17)": {
        "pdb": "sample_examples/receptors/1M17_receptor.pdb",
        "center": (22.014, 0.253, 52.794),
        "ligand": "Erlotinib (AQ4)",
        "source": "RCSB PDB 1M17; center is the bound ligand heavy-atom centroid",
    },
    "Abl kinase · imatinib / Gleevec (2HYY)": {
        "pdb": "sample_examples/receptors/2HYY_receptor.pdb",
        "center": (14.251, 15.283, 17.632),
        "ligand": "Imatinib (STI)",
        "source": "RCSB PDB 2HYY; center is the bound ligand heavy-atom centroid",
    },
    "HIV-1 protease · flap inhibitor (1HVR)": {
        "pdb": "sample_examples/receptors/1HVR_receptor.pdb",
        "center": (-9.192, 15.906, 27.946),
        "ligand": "XK263 inhibitor (XK2)",
        "source": "RCSB PDB 1HVR; center is the bound ligand heavy-atom centroid",
    },
    "Hsp90 chaperone · purine inhibitor (1UY6)": {
        "pdb": "sample_examples/receptors/1UY6_receptor.pdb",
        "center": (3.400, 11.180, 24.857),
        "ligand": "PU3 inhibitor (PU3)",
        "source": "RCSB PDB 1UY6; center is the bound ligand heavy-atom centroid",
    },
    "COX-2 synthase · indomethacin (4COX)": {
        "pdb": "sample_examples/receptors/4COX_receptor.pdb",
        "center": (24.849, 22.295, 15.126),
        "ligand": "Indomethacin (IMN)",
        "source": "RCSB PDB 4COX; center is the bound ligand heavy-atom centroid",
    },
}

st.subheader("1. Protein and pocket")
sample_name = st.selectbox("Choose a structure", list(SAMPLE_EXAMPLES), index=1)
sample = SAMPLE_EXAMPLES[sample_name]
if sample_name.startswith("EGFR"):
    st.warning("EGFR (1M17) can fail Gate Q in multi-state mode. The pipeline will retry and may fall back to 1 state; use FKBP12 (1FKB) for a true multi-state run.")

with st.form("pipeline_form"):
    pdb_upload = None
    if sample is None:
        pdb_upload = st.file_uploader("Protein structure (.pdb)", type=["pdb"])
        initial_center = (0.0, 0.0, 0.0)
        center_source = "User supplied; verify from a bound ligand or known binding site"
        pocket_cols = st.columns(4)
        with pocket_cols[0]:
            center_x = st.number_input("Pocket center X (Å)", value=initial_center[0], format="%.3f", key=f"center_x_{sample_name}")
        with pocket_cols[1]:
            center_y = st.number_input("Pocket center Y (Å)", value=initial_center[1], format="%.3f", key=f"center_y_{sample_name}")
        with pocket_cols[2]:
            center_z = st.number_input("Pocket center Z (Å)", value=initial_center[2], format="%.3f", key=f"center_z_{sample_name}")
        with pocket_cols[3]:
            pocket_radius = st.number_input("Pocket radius (Å)", min_value=3.0, max_value=25.0, value=10.0, step=1.0)
        pocket_note = st.text_input("Pocket center source", value=center_source)
        st.info("Enter the pocket center in the uploaded PDB's coordinate frame. The center of a known bound ligand is a useful starting point.")
    else:
        initial_center = sample["center"]
        center_source = sample["source"]
        center_x, center_y, center_z = initial_center
        pocket_radius = 10.0
        pocket_note = center_source
        st.info(f"Ready-to-run built-in example: {sample['ligand']}. The pocket center and receptor are filled in automatically.")
        st.caption(f"Pocket center: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f}) Å · radius: {pocket_radius:.1f} Å")

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
            value=sample is not None,
            help="Prepared examples may include residues without a Meeko template. Omitted residues can affect docking; inspect the run output before interpreting scores.",
        )
        st.caption("Default quick run: 2 protein states, 6 molecules, Vina exhaustiveness 1.")

    submitted = st.form_submit_button("Generate and dock molecules", type="primary", use_container_width=True)

if submitted:
    if sample is None and pdb_upload is None:
        st.error("Upload a protein PDB first.")
    else:
        try:
            uploaded_receptor = (
                _protein_receptor_bytes(pdb_upload.getvalue(), pdb_upload.name)
                if sample is None and pdb_upload is not None else None
            )
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
            if sample is None:
                pdb_path.write_bytes(uploaded_receptor)
            else:
                sample_pdb = _resolved_file(sample["pdb"])
                if not sample_pdb.is_file():
                    st.error(f"Built-in sample is missing: `{sample_pdb}`")
                    st.stop()
                try:
                    pdb_path.write_bytes(_protein_receptor_bytes(sample_pdb.read_bytes(), sample_pdb.name))
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
            command = _pipeline_command(
                pdb_path=pdb_path,
                out_dir=output_dir,
                center=(center_x, center_y, center_z),
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
