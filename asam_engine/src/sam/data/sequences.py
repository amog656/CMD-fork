aa_list = list("QWERTYIPASDFGHKLCVNM")
aa_one_letter = set(aa_list)

aa_three_letters = ["GLN", "TRP", "GLU", "ARG", "THR",
                    "TYR", "ILE", "PRO", "ALA", "SER",
                    "ASP", "PHE", "GLY", "HIS", "LYS",
                    "LEU", "CYS", "VAL", "ASN", "MET"]

aa_one_to_three_dict = {
    "G": "GLY", "A": "ALA", "L": "LEU", "I": "ILE", "R": "ARG", "K": "LYS",
    "M": "MET", "C": "CYS", "Y": "TYR", "T": "THR", "P": "PRO", "S": "SER",
    "W": "TRP", "D": "ASP", "E": "GLU", "N": "ASN", "Q": "GLN", "F": "PHE",
    "H": "HIS", "V": "VAL", "X": "XXX"
}

aa_three_to_one_dict = {
    "GLY": "G", "ALA": "A", "LEU": "L", "ILE": "I", "ARG": "R", "LYS": "K",
    "MET": "M", "CYS": "C", "TYR": "Y", "THR": "T", "PRO": "P", "SER": "S",
    "TRP": "W", "ASP": "D", "GLU": "E", "ASN": "N", "GLN": "Q", "PHE": "F",
    "HIS": "H", "VAL": "V", "XXX": "X"
}

gpcrdmd_three_letters = [
    "RET", "CSP", "UNK",
    "L00", "L01", "L02", "L03", "L04", "L05"
]
gpcrmd_alphabet_three_letters = aa_three_letters + gpcrdmd_three_letters


def get_num_beads(alphabet: str):
    if alphabet in (None, "standard"):
        return 20
    elif alphabet == "gpcrmd":
        return len(gpcrmd_alphabet_three_letters)
    else:
        raise KeyError(alphabet)

def check_alphabet(alphabet: str):
    if not alphabet in (None, "standard", "gpcrmd"):
        raise KeyError(alphabet)

def is_standard_alphabet(alphabet: str):
    return alphabet in (None, "standard")


pos_aa = set(("K", "R"))
neg_aa = set(("D", "E"))

def get_net_q_res(seq: str):
    q = 0
    for r in seq:
        if r in pos_aa:
            q += 1
        elif r in neg_aa:
            q -= 1
    return q/len(seq)


# Data from OpenFold.
ofo_restype_name_to_atom14_names = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "",
        "",
        "",
    ],
    "ASN": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "OD1",
        "ND2",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "ASP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "OD1",
        "OD2",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "NE2",
        "",
        "",
        "",
        "",
        "",
    ],
    "GLU": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "OE2",
        "",
        "",
        "",
        "",
        "",
    ],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "ND1",
        "CD2",
        "CE1",
        "NE2",
        "",
        "",
        "",
        "",
    ],
    "ILE": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG1",
        "CG2",
        "CD1",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "LEU": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "LYS": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "CE",
        "NZ",
        "",
        "",
        "",
        "",
        "",
    ],
    "MET": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "SD",
        "CE",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "PHE": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "",
        "",
        "",
    ],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "OG1",
        "CG2",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
        "",
        "",
    ],
    "VAL": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG1",
        "CG2",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
}

ofo_restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

chi_angles_atoms = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}