"""Small PDB IO helpers shared by file-oriented stages."""

from __future__ import annotations

from pathlib import Path


def read_pdb_atoms(path: str | Path) -> list[dict]:
    atoms = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 54:
                try:
                    atoms.append({
                        "line": line.rstrip("\n"), "record": line[:6].strip(), "atom": line[12:16].strip(), "resname": line[17:20].strip(),
                        "chain": line[21:22].strip(), "resid": line[22:26].strip(),
                        "element": line[76:78].strip() or line[12:16].strip()[0],
                        "xyz": [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    })
                except ValueError:
                    continue
    return atoms


def write_pdb_atoms(path: str | Path, atoms: list[dict]) -> None:
    out = []
    for index, atom in enumerate(atoms, start=1):
        x, y, z = atom["xyz"]
        out.append(f"ATOM  {index:5d} {atom['atom'][:4]:>4s} {atom['resname'][:3]:>3s} {atom['chain'][:1]:1s}{int(atom['resid']):4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {atom['element'][:2]:>2s}\n")
    out.append("END\n")
    Path(path).write_text("".join(out), encoding="utf-8")
