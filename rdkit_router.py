"""Enhanced RDKit API Router — extended cheminformatics functionality."""

from fastapi import APIRouter, HTTPException, Query, Depends, Response
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import math
import html as html_lib
import urllib.parse
import urllib.request

from database import get_db, CachedName
from Molecule import Molecule
from i18n import get_text, get_lang

import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Recap
from rdkit.Chem import QED, Draw, inchi
from rdkit.Chem import Lipinski, Crippen, MolSurf, GraphDescriptors
from rdkit.Chem import rdchem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdMolTransforms, rdMolAlign

router = APIRouter(prefix="/rdkit/api", tags=["RDKit Enhanced API"])

PROXY_ADDRESS = "127.0.0.1:9050"


@router.get("/", include_in_schema=False)
async def rdkit_index(lang: str = Depends(get_lang)):
    """RDKit API index page listing all endpoints."""
    endpoints = {
        get_text("rdkit_index_adme", lang): {
            "GET /rdkit/api/adme": get_text("ep_adme_desc", lang),
        },
        get_text("rdkit_index_analysis", lang): {
            "GET /rdkit/api/functional_groups": get_text("ep_functional_groups_desc", lang),
            "GET /rdkit/api/validate": get_text("ep_validate_desc", lang),
            "GET /rdkit/api/pka": get_text("ep_pka_desc", lang),
            "GET /rdkit/api/descriptors": get_text("ep_descriptors_desc", lang),
            "GET /rdkit/api/huckel": get_text("ep_hybridization_desc", lang),
        },
        get_text("rdkit_index_reactions", lang): {
            "GET /rdkit/api/reaction": get_text("ep_reaction_desc", lang),
            "GET /rdkit/api/by_inchikey": get_text("ep_by_inchikey_desc", lang),
        },
        get_text("rdkit_index_3d", lang): {
            "GET /rdkit/api/3d": get_text("ep_3d_desc", lang),
            "GET /rdkit/api/render_highlighted": get_text("ep_render_highlighted_desc", lang),
        },
        get_text("rdkit_index_batch", lang): {
            "POST /rdkit/api/batch": get_text("ep_batch_desc", lang),
            "GET /rdkit/api/library": get_text("ep_library_desc", lang),
            "POST /rdkit/api/library/save": get_text("ep_library_save_desc", lang),
            "DELETE /rdkit/api/library/{smiles}": get_text("ep_library_delete_desc", lang),
        },
        get_text("rdkit_index_comparison", lang): {
            "POST /rdkit/api/compare": get_text("ep_compare_desc", lang),
        },
    }
    return {"endpoints": endpoints, "total": sum(len(v) for v in endpoints.values())}


def _optimize_and_energy(mol, conf_id):
    """Optimize a conformer with MMFF, falling back to UFF when parameters are unavailable. Returns the energy in kcal/mol."""
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=2000)
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    else:
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=2000)
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
    return ff.CalcEnergy()


def set_proxy(proxy: str):
    global PROXY_ADDRESS
    PROXY_ADDRESS = proxy


# ─────────────────────────────────────────────────────────────────────────────
# ADME / Drug-likeness
# ─────────────────────────────────────────────────────────────────────────────


class ADMEResponse(BaseModel):
    smiles: str
    formula: str
    molecular_weight: float
    logp: float
    hbd: int
    hba: int
    tpsa: float
    rotatable_bonds: int
    lipinski_violations: int
    lipinski_pass: bool
    qed: float
    qed_interpretation: str
    sa_score: Optional[float] = None
    num_rings: int
    num_aromatic_rings: int
    num_heteroatoms: int
    num_atoms: int
    num_bonds: int
    fraction_sp3: float
    num_stereocenters: int
    num_undefined_stereocenters: int


@router.get(
    "/adme", response_model=ADMEResponse, summary=get_text("ep_adme_desc")
)
def get_adme_properties(
    smiles: str = Query(..., description=get_text("param_smiles")),
    lang: str = Depends(get_lang),
):
    """
    Full set of ADME properties and drug-likeness scores:
    - Lipinski Rule of Five
    - QED (Quantitative Estimate of Drug-likeness)
    - SA Score (Synthetic Accessibility)
    - Basic descriptors
    """
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        mol = m.m

        # Lipinski's Rule of Five
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1

        # QED
        qed_score = QED.qed(mol)
        if qed_score >= 0.7:
            qed_interp = get_text("qed_excellent", lang)
        elif qed_score >= 0.5:
            qed_interp = get_text("qed_good", lang)
        elif qed_score >= 0.3:
            qed_interp = get_text("qed_fair", lang)
        else:
            qed_interp = get_text("qed_poor", lang)

        # SA Score (synthetic accessibility)
        try:
            from rdkit.Contrib.SA_Score import sascorer

            sa = sascorer.calculateScore(mol)
        except ImportError:
            sa = None

        # Stereocenters
        stereo = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        defined = Chem.FindMolChiralCenters(mol, includeUnassigned=False)

        return ADMEResponse(
            smiles=smiles,
            formula=rdMolDescriptors.CalcMolFormula(mol),
            molecular_weight=round(mw, 4),
            logp=round(logp, 4),
            hbd=hbd,
            hba=hba,
            tpsa=round(Descriptors.TPSA(mol), 4),
            rotatable_bonds=Descriptors.NumRotatableBonds(mol),
            lipinski_violations=violations,
            lipinski_pass=(violations <= 1),
            qed=round(qed_score, 4),
            qed_interpretation=qed_interp,
            sa_score=round(sa, 2) if sa is not None else None,
            num_rings=rdMolDescriptors.CalcNumRings(mol),
            num_aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
            num_heteroatoms=rdMolDescriptors.CalcNumHeteroatoms(mol),
            num_atoms=mol.GetNumAtoms(),
            num_bonds=mol.GetNumBonds(),
            fraction_sp3=round(Lipinski.FractionCSP3(mol), 4),
            num_stereocenters=len(stereo),
            num_undefined_stereocenters=len(stereo) - len(defined),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Функциональные группы
# ─────────────────────────────────────────────────────────────────────────────

FUNCTIONAL_GROUPS = {
    "hydroxyl": {"smarts": "[OX2H]", "name": "Hydroxyl (-OH)"},
    "carboxylic_acid": {
        "smarts": "[CX3](=O)[OX2H1]",
        "name": "Carboxylic acid (-COOH)",
    },
    "amine_primary": {"smarts": "[NX3;H2;!$(NC=O)]", "name": "Primary amine (-NH2)"},
    "amine_secondary": {
        "smarts": "[NX3;H1;!$(NC=O)]",
        "name": "Secondary amine (-NH-)",
    },
    "amine_tertiary": {"smarts": "[NX3;H0;!$(NC=O)]", "name": "Tertiary amine (-N<)"},
    "amide": {"smarts": "[NX3][CX3](=[OX1])", "name": "Amide (-C(=O)N-)"},
    "nitro": {"smarts": "[$([NX3+](=O)[O-]),$([NX3](=O)=O)]", "name": "Nitro (-NO2)"},
    "sulfonyl": {"smarts": "[SX4](=[OX1])(=[OX1])", "name": "Sulfonyl (-SO2-)"},
    "phosphate": {"smarts": "[PX4](=[OX1])([OX2])[OX2]", "name": "Phosphate"},
    "aldehyde": {"smarts": "[CX3H1](=O)[#6]", "name": "Aldehyde (-CHO)"},
    "ketone": {"smarts": "[#6][CX3](=O)[#6]", "name": "Ketone (>C=O)"},
    "ester": {"smarts": "[#6][CX3](=O)[OX2H0][#6]", "name": "Ester (-COOR)"},
    "ether": {"smarts": "[OD2]([#6])[#6]", "name": "Ether (-O-)"},
    "alkene": {"smarts": "[CX3]=[CX3]", "name": "Alkene (C=C)"},
    "alkyne": {"smarts": "[CX2]#[CX2]", "name": "Alkyne (C≡C)"},
    "aromatic_ring": {"smarts": "c1ccccc1", "name": "Aromatic ring (benzene)"},
    "heterocycle": {"smarts": "[!#6]1~[!#6]~*~*~*~1", "name": "Heterocycle"},
    "halogen_f": {"smarts": "[F]", "name": "Fluorine"},
    "halogen_cl": {"smarts": "[Cl]", "name": "Chlorine"},
    "halogen_br": {"smarts": "[Br]", "name": "Bromine"},
    "halogen_i": {"smarts": "[I]", "name": "Iodine"},
    "thiol": {"smarts": "[SX2H]", "name": "Thiol (-SH)"},
    "nitrile": {"smarts": "[CX2]#[NX1]", "name": "Nitrile (-C≡N)"},
    "azide": {"smarts": "[NX2]=[NX2]=[NX1]", "name": "Azide (-N3)"},
    "imine": {"smarts": "[CX3]=[NX2]", "name": "Imine (C=N)"},
    "hydrazine": {"smarts": "[NX3][NX3]", "name": "Hydrazine (-N-N-)"},
    "urea": {"smarts": "[NX3][CX3](=[OX1])[NX3]", "name": "Urea"},
    "carbamate": {"smarts": "[OX2][CX3](=[OX1])[NX3]", "name": "Carbamate"},
    "sulfonamide": {"smarts": "[SX4](=[OX1])(=[OX1])[NX3]", "name": "Sulfonamide"},
}


@router.get("/functional_groups", summary=get_text("ep_functional_groups_desc"))
def find_functional_groups(
    smiles: str = Query(..., description=get_text("param_smiles")),
    lang: str = Depends(get_lang),
):
    """
    Detect functional groups in a molecule using SMARTS patterns.
    Returns the matched groups together with their counts.
    """
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        mol = m.m
        found = []
        for key, info in FUNCTIONAL_GROUPS.items():
            pattern = Chem.MolFromSmarts(info["smarts"])
            if pattern is None:
                continue
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                found.append(
                    {
                        "key": key,
                        "name": get_text(f"fg_{key}", lang),
                        "smarts": info["smarts"],
                        "count": len(matches),
                        "atom_indices": [list(m) for m in matches],
                    }
                )

        return {
            "smiles": smiles,
            "total_groups_found": len(found),
            "functional_groups": found,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Валидация SMILES с диагностикой
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/validate", summary=get_text("ep_validate_desc"))
def validate_smiles(
    smiles: str = Query(..., description=get_text("param_smiles_to_validate")),
    lang: str = Depends(get_lang),
):
    """
    Validate a SMILES string and return detailed diagnostics:
    - error type (valence, aromaticity, syntax)
    - error position
    - suggested fixes
    """
    result = {
        "smiles": smiles,
        "valid": False,
        "canonical_smiles": None,
        "error_type": None,
        "error_message": None,
        "error_position": None,
        "warnings": [],
    }

    if not smiles or not smiles.strip():
        result["error_type"] = "empty"
        result["error_message"] = get_text("val_empty", lang)
        return result

    smiles = smiles.strip()

    # Проверка на недопустимые символы
    allowed_chars = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]=#@+-./\\:%*"
    )
    invalid_chars = set()
    for i, ch in enumerate(smiles):
        if ch not in allowed_chars:
            invalid_chars.add((i, ch))
    if invalid_chars:
        result["error_type"] = "invalid_characters"
        result["error_message"] = f"{get_text('val_invalid_characters', lang)}: {invalid_chars}"
        result["error_position"] = min(invalid_chars)[0]
        return result

    # Проверка баланса скобок
    open_paren = smiles.count("(")
    close_paren = smiles.count(")")
    if open_paren != close_paren:
        result["error_type"] = "unbalanced_parentheses"
        result["error_message"] = (
            f"{get_text('val_unbalanced_parentheses', lang)}: {open_paren} open vs {close_paren} close"
        )
        return result

    open_bracket = smiles.count("[")
    close_bracket = smiles.count("]")
    if open_bracket != close_bracket:
        result["error_type"] = "unbalanced_brackets"
        result["error_message"] = (
            f"{get_text('val_unbalanced_brackets', lang)}: {open_bracket} open vs {close_bracket} close"
        )
        return result

    # Попытка парсинга
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Попытка получить более детальную ошибку
        try:
            from rdkit.Chem import rdMolDescriptors

            # Пробуем с sanitize=False
            mol_raw = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol_raw is not None:
                try:
                    Chem.SanitizeMol(mol_raw)
                except Exception as sanitize_err:
                    err_str = str(sanitize_err)
                    if "valence" in err_str.lower():
                        result["error_type"] = "valence_error"
                    elif "aromatic" in err_str.lower():
                        result["error_type"] = "aromaticity_error"
                    elif "kekul" in err_str.lower():
                        result["error_type"] = "kekulization_error"
                    else:
                        result["error_type"] = "sanitization_error"
                    result["error_message"] = err_str
            else:
                result["error_type"] = "syntax_error"
                result["error_message"] = get_text("val_syntax_error", lang)
        except Exception:
            result["error_type"] = "syntax_error"
            result["error_message"] = get_text("val_syntax_error", lang)
        return result

    # Валидный SMILES
    result["valid"] = True
    result["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)

    # Предупреждения
    if mol.GetNumAtoms() > 100:
        result["warnings"].append(get_text("val_large_molecule", lang))
    if Descriptors.MolWt(mol) > 1000:
        result["warnings"].append(get_text("val_high_mw", lang))
    if Descriptors.MolLogP(mol) > 7:
        result["warnings"].append(get_text("val_high_logp", lang))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Реакции (SMIRKS)
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/reaction", summary=get_text("ep_reaction_desc"))
def apply_reaction(
    smiles: str = Query(..., description=get_text("param_smiles_reagent")),
    reaction_smarts: str = Query(..., description=get_text("param_reaction_smarts")),
    lang: str = Depends(get_lang),
):
    """
    Apply a reaction (SMIRKS) to a molecule.
    Returns the list of reaction products.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(get_text("invalid_smiles", lang))

        rxn = AllChem.ReactionFromSmarts(reaction_smarts)
        if rxn is None:
            raise ValueError(get_text("invalid_smirks_reaction", lang))

        products = rxn.RunReactants((mol,))

        if not products:
            return {
                "smiles": smiles,
                "reaction_smarts": reaction_smarts,
                "products": [],
                "product_count": 0,
                "message": get_text("no_reaction_products", lang),
            }

        unique_products = []
        seen = set()
        for product_set in products:
            for product in product_set:
                try:
                    Chem.SanitizeMol(product)
                    smi = Chem.MolToSmiles(product, canonical=True)
                    if smi not in seen:
                        seen.add(smi)
                        unique_products.append(smi)
                except Exception:
                    continue

        return {
            "smiles": smiles,
            "reaction_smarts": reaction_smarts,
            "products": unique_products,
            "product_count": len(unique_products),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# pKa (оценка)
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/pka", summary=get_text("ep_pka_desc"))
def estimate_pka(
    smiles: str = Query(..., description=get_text("param_smiles")),
    lang: str = Depends(get_lang),
):
    """
    Estimate the acidic and basic centres of a molecule.
    Uses a heuristic approach based on functional groups.
    """
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        mol = m.m
        pka_sites = []

        # Карбоновые кислоты (pKa ~4-5)
        acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
        if acid_pattern:
            for match in mol.GetSubstructMatches(acid_pattern):
                pka_sites.append(
                    {
                        "atom_idx": match[0],
                        "type": "carboxylic_acid",
                        "type_name": get_text("pka_carboxylic_acid", lang),
                        "estimated_pka": 4.5,
                        "strength": get_text("pka_acidic", lang),
                    }
                )

        # Фенолы (pKa ~9-10)
        phenol_pattern = Chem.MolFromSmarts("[OX2H]c1ccccc1")
        if phenol_pattern:
            for match in mol.GetSubstructMatches(phenol_pattern):
                pka_sites.append(
                    {
                        "atom_idx": match[0],
                        "type": "phenol",
                        "type_name": get_text("pka_phenol", lang),
                        "estimated_pka": 10.0,
                        "strength": get_text("pka_acidic", lang),
                    }
                )

        # Алифатические амины (pKa ~9-11)
        amine_pattern = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O);!$([NH2]c1ccccc1)]")
        if amine_pattern:
            for match in mol.GetSubstructMatches(amine_pattern):
                pka_sites.append(
                    {
                        "atom_idx": match[0],
                        "type": "aliphatic_amine",
                        "type_name": get_text("pka_aliphatic_amine", lang),
                        "estimated_pka": 10.0,
                        "strength": get_text("pka_basic", lang),
                    }
                )

        # Anilines (pKa ~4-5)
        aniline_pattern = Chem.MolFromSmarts("[NX3;H2,H1]c1ccccc1")
        if aniline_pattern:
            for match in mol.GetSubstructMatches(aniline_pattern):
                pka_sites.append(
                    {
                        "atom_idx": match[0],
                        "type": "aniline",
                        "type_name": get_text("pka_aniline", lang),
                        "estimated_pka": 4.6,
                        "strength": get_text("pka_basic", lang),
                    }
                )

        # Тиолы (pKa ~8-10)
        thiol_pattern = Chem.MolFromSmarts("[SX2H]")
        if thiol_pattern:
            for match in mol.GetSubstructMatches(thiol_pattern):
                pka_sites.append(
                    {
                        "atom_idx": match[0],
                        "type": "thiol",
                        "type_name": get_text("pka_thiol", lang),
                        "estimated_pka": 9.0,
                        "strength": get_text("pka_acidic", lang),
                    }
                )

        # Sulfonic acids (pKa ~-2)
        sulfonic_pattern = Chem.MolFromSmarts("[SX4](=[OX1])(=[OX1])[OX2H1]")
        if sulfonic_pattern:
            for match in mol.GetSubstructMatches(sulfonic_pattern):
                pka_sites.append(
                    {
                        "atom_idx": match[0],
                        "type": "sulfonic_acid",
                        "type_name": get_text("pka_sulfonic_acid", lang),
                        "estimated_pka": -2.0,
                        "strength": get_text("pka_strong_acid", lang),
                    }
                )

        return {
            "smiles": smiles,
            "pka_sites": pka_sites,
            "total_sites": len(pka_sites),
            "acidic_sites": len(
                [s for s in pka_sites if s["strength"] in (get_text("pka_acidic", lang), get_text("pka_strong_acid", lang))]
            ),
            "basic_sites": len([s for s in pka_sites if s["strength"] == get_text("pka_basic", lang)]),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Batch API
# ─────────────────────────────────────────────────────────────────────────────


class BatchRequest(BaseModel):
    smiles_list: List[str]
    properties: Optional[List[str]] = None  # Если None — все свойства


@router.post("/batch", summary=get_text("ep_batch_desc"))
def batch_analyze(request: BatchRequest, lang: str = Depends(get_lang)):
    """
    Analyze a list of SMILES strings in a single request.
    Returns the properties of every molecule.
    """
    results = []
    for smiles in request.smiles_list:
        try:
            m = Molecule(smiles)
            if m.m is None:
                results.append(
                    {"smiles": smiles, "valid": False, "error": get_text("invalid_smiles", lang)}
                )
                continue

            mol = m.m
            props = {
                "smiles": smiles,
                "valid": True,
                "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
                "formula": rdMolDescriptors.CalcMolFormula(mol),
                "molecular_weight": round(Descriptors.MolWt(mol), 4),
                "logp": round(Descriptors.MolLogP(mol), 4),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "tpsa": round(Descriptors.TPSA(mol), 4),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "qed": round(QED.qed(mol), 4),
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "num_rings": rdMolDescriptors.CalcNumRings(mol),
            }

            # Lipinski
            violations = 0
            if props["molecular_weight"] > 500:
                violations += 1
            if props["logp"] > 5:
                violations += 1
            if props["hbd"] > 5:
                violations += 1
            if props["hba"] > 10:
                violations += 1
            props["lipinski_violations"] = violations
            props["lipinski_pass"] = violations <= 1

            results.append(props)
        except Exception as e:
            results.append({"smiles": smiles, "valid": False, "error": str(e)})

    return {
        "total": len(request.smiles_list),
        "valid_count": sum(1 for r in results if r.get("valid")),
        "invalid_count": sum(1 for r in results if not r.get("valid")),
        "results": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# InChIKey поддержка
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/by_inchikey", summary=get_text("ep_by_inchikey_desc"))
def get_by_inchikey(
    inchikey: str = Query(..., description=get_text("param_inchikey")),
    lang: str = Depends(get_lang),
):
    """
    Retrieve molecule properties by InChIKey.
    Uses the Cactus/NCI resolver for conversion.
    """
    try:
        # Проверка формата InChIKey
        if len(inchikey) != 27 or inchikey.count("-") != 2:
            raise ValueError(get_text("invalid_inchikey_format", lang))

        # Попытка разрешить через Cactus
        safe_key = urllib.parse.quote(inchikey, safe="")
        target_url = f"https://cactus.nci.nih.gov/chemical/structure/{safe_key}/smiles"

        opener = urllib.request.build_opener()
        if PROXY_ADDRESS.strip():
            proxy_host, proxy_port = PROXY_ADDRESS.split(":")
            import socks
            from sockshandler import SocksiPyHandler

            proxy_handler = SocksiPyHandler(socks.SOCKS5, proxy_host, int(proxy_port))
            opener = urllib.request.build_opener(proxy_handler)

        req = urllib.request.Request(target_url, headers={"User-Agent": "Mozilla/5.0"})
        with opener.open(req, timeout=10) as resp:
            smiles = resp.read().decode("utf-8").strip()

        if not smiles:
            raise ValueError(get_text("could_not_resolve_inchikey", lang))

        # Получаем свойства
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("resolved_smiles_invalid", lang))

        mol = m.m
        return {
            "inchikey": inchikey,
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "molecular_weight": round(Descriptors.MolWt(mol), 4),
            "logp": round(Descriptors.MolLogP(mol), 4),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "tpsa": round(Descriptors.TPSA(mol), 4),
            "qed": round(QED.qed(mol), 4),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Полный набор дескрипторов
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/descriptors", summary=get_text("ep_descriptors_desc"))
def get_all_descriptors(
    smiles: str = Query(..., description=get_text("param_smiles")),
    lang: str = Depends(get_lang),
):
    """
    Return every available RDKit descriptor for a molecule.
    Includes more than 200 molecular descriptors.
    """
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        mol = m.m

        # Вычисляем все дескрипторы
        descriptors = {}
        for name, func in Descriptors.descList:
            try:
                value = func(mol)
                if isinstance(value, (int, float)):
                    if isinstance(value, float) and not math.isfinite(value):
                        descriptors[name] = None
                    else:
                        descriptors[name] = (
                            round(value, 6) if isinstance(value, float) else value
                        )
            except Exception:
                descriptors[name] = None

        # Дополнительные дескрипторы
        extra = {
            "NumAtoms": mol.GetNumAtoms(),
            "NumBonds": mol.GetNumBonds(),
            "NumHeavyAtoms": mol.GetNumHeavyAtoms(),
            "NumRings": rdMolDescriptors.CalcNumRings(mol),
            "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
            "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
            "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "FractionCSP3": Lipinski.FractionCSP3(mol),
            "MolWt": Descriptors.MolWt(mol),
            "ExactMolWt": Descriptors.ExactMolWt(mol),
            "MolLogP": Descriptors.MolLogP(mol),
            "MolMR": Descriptors.MolMR(mol),
            "TPSA": Descriptors.TPSA(mol),
            "BalabanJ": GraphDescriptors.BalabanJ(mol),
            "BertzCT": GraphDescriptors.BertzCT(mol),
            "Chi0": GraphDescriptors.Chi0(mol),
            "Chi1": GraphDescriptors.Chi1(mol),
            "Kappa1": GraphDescriptors.Kappa1(mol),
            "Kappa2": GraphDescriptors.Kappa2(mol),
            "Kappa3": GraphDescriptors.Kappa3(mol),
            "HallKierAlpha": Descriptors.HallKierAlpha(mol),
            "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
            "NumRadicalElectrons": Descriptors.NumRadicalElectrons(mol),
            "MaxPartialCharge": Descriptors.MaxPartialCharge(mol),
            "MinPartialCharge": Descriptors.MinPartialCharge(mol),
            "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge(mol),
            "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge(mol),
        }

        # QED components
        try:
            qed_props = QED.properties(mol)
            extra["QED"] = {
                "MW": round(qed_props.MW, 4),
                "ALOGP": round(qed_props.ALOGP, 4),
                "HBA": qed_props.HBA,
                "HBD": qed_props.HBD,
                "PSA": round(qed_props.PSA, 4),
                "ROTB": qed_props.ROTB,
                "AROM": qed_props.AROM,
                "ALERTS": qed_props.ALERTS,
                "score": round(QED.qed(mol), 4),
            }
        except Exception:
            pass

        extra = {
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in extra.items()
        }

        return {
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
            "descriptor_count": len(descriptors) + len(extra),
            "descriptors": descriptors,
            "extra_descriptors": extra,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Гибридизация, π-электроны и ароматичность по правилу Хюккеля
# ─────────────────────────────────────────────────────────────────────────────

# Число валентных электронов для основных элементов (для расчёта
# неподелённых пар и свободных p-орбиталей).
_VALENCE_ELECTRONS = {
    1: 1,
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    9: 7,
    14: 4,
    15: 5,
    16: 6,
    17: 7,
    31: 3,
    32: 4,
    33: 5,
    34: 6,
    35: 7,
    49: 3,
    50: 4,
    51: 5,
    52: 6,
    53: 7,
    81: 3,
    82: 4,
    83: 5,
    84: 6,
    85: 7,
}

_HYBRIDIZATION_LABELS = {
    rdchem.HybridizationType.S: "s",
    rdchem.HybridizationType.SP: "sp",
    rdchem.HybridizationType.SP2: "sp2",
    rdchem.HybridizationType.SP2D: "sp2d",
    rdchem.HybridizationType.SP3: "sp3",
    rdchem.HybridizationType.SP3D: "sp3d",
    rdchem.HybridizationType.SP3D2: "sp3d2",
    rdchem.HybridizationType.OTHER: "other",
    rdchem.HybridizationType.UNSPECIFIED: "unspecified",
}


def _hybridization_label(atom) -> str:
    """Conventional label for the hybridization of an atom (sp, sp2, sp3, ...)."""
    hybrid = atom.GetHybridization()
    # RDKit иногда помечает ароматический бор как sp из-за валентности.
    # В ароматическом цикле такой атом фактически sp².
    if (
        atom.GetIsAromatic()
        and hybrid == rdchem.HybridizationType.SP
        and not any(b.GetBondType() == Chem.BondType.TRIPLE for b in atom.GetBonds())
        and atom.GetDegree() >= 2
    ):
        return "sp2"
    return _HYBRIDIZATION_LABELS.get(hybrid, "unspecified")


def _atom_total_valence(atom) -> int:
    """Total valence of an atom, including implicit hydrogens."""
    try:
        explicit = int(atom.GetValence(rdchem.ValenceType.EXPLICIT))
        return explicit + int(atom.GetNumImplicitHs())
    except Exception:
        try:
            return int(atom.GetTotalValence())
        except Exception:
            return 0


def _atom_lone_pairs(atom):
    """Estimate the number of lone electron pairs on an atom."""
    valence_electrons = _VALENCE_ELECTRONS.get(atom.GetAtomicNum())
    if valence_electrons is None:
        return None
    lone_pairs = (
        valence_electrons - int(atom.GetFormalCharge()) - _atom_total_valence(atom)
    ) / 2.0
    if lone_pairs < 0:
        return 0
    rounded = round(lone_pairs)
    return int(rounded) if abs(lone_pairs - rounded) < 1e-6 else round(lone_pairs, 2)


def _atom_has_multiple_bond(atom) -> bool:
    """Whether the atom has a multiple bond (including aromatic bonds)."""
    for bond in atom.GetBonds():
        if bond.GetIsAromatic() or bond.GetBondType() in (
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
        ):
            return True
    return False


def _atom_has_p_orbital(atom) -> bool:
    """Whether the atom can participate in a pi system through a p orbital."""
    if atom.GetIsAromatic():
        return True
    if atom.GetHybridization() in (
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
    ):
        return True
    return _atom_has_multiple_bond(atom)


def _kekulized_copy(mol):
    """Copy of the molecule with explicit double bonds (for pi-electron counting)."""
    kekulized = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekulized, clearAromaticFlags=True)
        return kekulized, True
    except Exception:
        # Для нестандартных ароматических систем оставляем исходные связи.
        return Chem.Mol(mol), False


def _has_internal_multiple_bond(kekulized_mol, atom_idx: int, atom_set) -> bool:
    """Whether the atom has a multiple bond to another atom in the same ring."""
    try:
        atom = kekulized_mol.GetAtomWithIdx(atom_idx)
    except Exception:
        return False
    for bond in atom.GetBonds():
        if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            if bond.GetOtherAtomIdx(atom_idx) in atom_set:
                return True
    return False


def _atom_pi_contribution(atom, kekulized_mol, atom_set):
    """
    Contribution of an atom to the pi system (0, 1 or 2 electrons).

    Returns (number of pi electrons, p-orbital state code):
      bond        — the atom takes part in a multiple bond (p1);
      lone_pair   — a lone pair occupies the p orbital (p2);
      radical     — an unpaired electron (p1);
      empty       — an empty p orbital (p0);
      exocyclic   — the p orbital is used by an exocyclic multiple bond;
      none        — no p orbital / the atom does not participate.
    """
    if not _atom_has_p_orbital(atom):
        return 0, "none"

    if _has_internal_multiple_bond(kekulized_mol, atom.GetIdx(), atom_set):
        return 1, "bond"

    if atom.GetNumRadicalElectrons() > 0:
        return 1, "radical"

    lone_pairs = _atom_lone_pairs(atom)
    if lone_pairs is not None and lone_pairs > 0 and atom.GetFormalCharge() <= 0:
        # Углерод/бор с неподелённой парой участвует в π-системе только
        # если он sp²/sp (карбанион, карбен), а не насыщенный sp³-атом.
        if atom.GetAtomicNum() in (5, 6):
            if atom.GetHybridization() in (
                rdchem.HybridizationType.SP,
                rdchem.HybridizationType.SP2,
            ) or atom.GetIsAromatic():
                return 2, "lone_pair"
            return 0, "none"
        return 2, "lone_pair"

    if atom.GetFormalCharge() > 0:
        return 0, "empty"

    # p-Орбиталь занята экзоциклической кратной связью,
    # но в π-систему самого цикла электрон не отдаёт.
    for bond in atom.GetBonds():
        if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            return 0, "exocyclic"

    return 0, "none"


def _ring_systems(mol):
    """
    Merge SSSR rings into ring systems (fused rings stay together).
    Returns a list of atom-index sets.
    """
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    if not rings:
        return []

    parent = list(range(len(rings)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if rings[i] & rings[j]:
                union(i, j)

    groups = {}
    for i, ring in enumerate(rings):
        groups.setdefault(find(i), set()).update(ring)

    return sorted(groups.values(), key=lambda atoms: min(atoms))


def _huckel_status(pi_electrons: int, conjugated: bool) -> dict:
    """Check the simple Huckel 4n+2 rule for a ring system."""
    if not conjugated:
        return {
            "is_huckel_aromatic": False,
            "status": "nonconjugated",
            "n": None,
            "formula": "not conjugated",
        }
    if pi_electrons >= 2 and pi_electrons % 4 == 2:
        n = (pi_electrons - 2) // 4
        return {
            "is_huckel_aromatic": True,
            "status": "aromatic",
            "n": n,
            "formula": f"4n+2 (n={n})",
        }
    if pi_electrons > 0 and pi_electrons % 4 == 0:
        n = pi_electrons // 4
        return {
            "is_huckel_aromatic": False,
            "status": "antiaromatic",
            "n": n,
            "formula": f"4n (n={n})",
        }
    return {
        "is_huckel_aromatic": False,
        "status": "nonaromatic",
        "n": None,
        "formula": "not 4n+2",
    }


@router.get(
    "/huckel",
    summary=get_text("ep_hybridization_desc"),
)
def analyze_huckel_aromaticity(
    smiles: str = Query(..., description=get_text("param_smiles")),
    lang: str = Depends(get_lang),
):
    """
    Analyze atom hybridization, pi electrons and ring aromaticity.

    For every heavy atom, return its hybridization (sp/sp2/sp3, ...), formal
    charge, number of lone pairs, pi-system contribution and p-orbital state.
    For every ring system (fused rings are merged), count the pi electrons and
    check the Huckel 4n+2 rule. RDKit aromaticity is reported as well.

    **Example request:**
    ```
    GET /rdkit/api/huckel?smiles=c1cc[nH]c1&lang=en
    ```
    """
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        mol = m.m
        kekulized_mol, kekulized_ok = _kekulized_copy(mol)
        systems = _ring_systems(mol)
        all_heavy_atoms = {
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1
        }

        atom_to_system = {}
        for system in systems:
            for atom_idx in system:
                atom_to_system[atom_idx] = system

        atoms_data = []
        atom_pi_electrons = {}
        hybridization_counts = {}
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue

            atom_idx = atom.GetIdx()
            system = atom_to_system.get(atom_idx)
            # Для атомов вне циклов считаем локальный вклад в π-систему
            # (например, C=O, C=C, неподелённая пара амина).
            contribution_set = system if system is not None else all_heavy_atoms
            pi_electrons, p_orbital_status = _atom_pi_contribution(
                atom, kekulized_mol, contribution_set
            )
            atom_pi_electrons[atom_idx] = pi_electrons

            hybridization = _hybridization_label(atom)
            hybridization_counts[hybridization] = (
                hybridization_counts.get(hybridization, 0) + 1
            )

            atoms_data.append(
                {
                    "index": atom_idx,
                    "element": atom.GetSymbol(),
                    "hybridization": hybridization,
                    "formal_charge": int(atom.GetFormalCharge()),
                    "lone_pairs": _atom_lone_pairs(atom),
                    "radical_electrons": int(atom.GetNumRadicalElectrons()),
                    "is_aromatic": bool(atom.GetIsAromatic()),
                    "in_ring": bool(atom.IsInRing()),
                    "pi_electrons": int(pi_electrons),
                    "p_orbital_status": p_orbital_status,
                }
            )

        ring_systems_data = []
        for system_index, system in enumerate(systems, start=1):
            pi_electrons = sum(atom_pi_electrons.get(idx, 0) for idx in system)
            conjugated = all(
                _atom_has_p_orbital(mol.GetAtomWithIdx(idx)) for idx in system
            )
            huckel = _huckel_status(pi_electrons, conjugated)

            internal_bonds = [
                bond
                for bond in mol.GetBonds()
                if bond.GetBeginAtomIdx() in system and bond.GetEndAtomIdx() in system
            ]
            rdkit_aromatic = bool(internal_bonds) and all(
                mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in system
            ) and all(bond.GetIsAromatic() for bond in internal_bonds)

            heteroatoms = [
                {"index": idx, "element": mol.GetAtomWithIdx(idx).GetSymbol()}
                for idx in sorted(system)
                if mol.GetAtomWithIdx(idx).GetAtomicNum() not in (1, 6)
            ]
            sssr_count = sum(1 for ring in mol.GetRingInfo().AtomRings() if set(ring) & system)

            ring_systems_data.append(
                {
                    "id": system_index,
                    "atom_indices": sorted(system),
                    "size": len(system),
                    "elements": sorted(
                        {mol.GetAtomWithIdx(idx).GetSymbol() for idx in system}
                    ),
                    "heterocycle": bool(heteroatoms),
                    "heteroatoms": heteroatoms,
                    "pi_electrons": int(pi_electrons),
                    "is_conjugated": bool(conjugated),
                    "kekulized": bool(kekulized_ok),
                    "polycyclic": sssr_count > 1,
                    "huckel": huckel,
                    "rdkit_aromatic": rdkit_aromatic,
                }
            )

        aromatic_ring_systems = sum(
            1 for system in ring_systems_data if system["rdkit_aromatic"]
        )
        huckel_aromatic_ring_systems = sum(
            1
            for system in ring_systems_data
            if system["huckel"]["is_huckel_aromatic"]
        )

        return {
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
            "summary": {
                "total_heavy_atoms": len(all_heavy_atoms),
                "hybridization_counts": hybridization_counts,
                "total_ring_systems": len(ring_systems_data),
                "aromatic_ring_systems": aromatic_ring_systems,
                "huckel_aromatic_ring_systems": huckel_aromatic_ring_systems,
                "total_ring_pi_electrons": sum(
                    system["pi_electrons"] for system in ring_systems_data
                ),
                "has_heterocycle": any(
                    system["heterocycle"] for system in ring_systems_data
                ),
                "rdkit_aromatic": aromatic_ring_systems > 0,
            },
            "atoms": atoms_data,
            "ring_systems": ring_systems_data,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 3D визуализация (base64 SDF + координаты)
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/3d", summary=get_text("ep_3d_desc"))
def get_3d_structure(
    smiles: str = Query(..., description=get_text("param_smiles")),
    format: str = Query("sdf", description=get_text("param_format")),
    lang: str = Depends(get_lang),
):
    """
    Generate the 3D structure of a molecule.
    Returns it in SDF, MOL or PDB format.
    """
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        mol = Chem.AddHs(m.m)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == -1:
            raise ValueError(get_text("failed_to_embed_molecule", lang))
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            AllChem.UFFOptimizeMolecule(mol)

        if format.lower() == "sdf":
            block = Chem.MolToMolBlock(mol)
            media_type = "chemical/x-mdl-sdfile"
        elif format.lower() == "mol":
            block = Chem.MolToMolBlock(mol)
            media_type = "chemical/x-mdl-molfile"
        elif format.lower() == "pdb":
            block = Chem.MolToPDBBlock(mol)
            media_type = "chemical/x-pdb"
        else:
            raise ValueError(f"{get_text('unsupported_format', lang)}: {format}")

        return Response(content=block, media_type=media_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# История / Библиотека молекул
# ─────────────────────────────────────────────────────────────────────────────


class MoleculeEntry(BaseModel):
    smiles: str
    name: Optional[str] = None
    tags: Optional[List[str]] = None


@router.post("/library/save", summary=get_text("ep_library_save_desc"))
def save_to_library(
    entry: MoleculeEntry,
    lang: str = Depends(get_lang),
    db: Session = Depends(get_db),
):
    """Save a molecule to the local library (SQLite)."""
    try:
        m = Molecule(entry.smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        canonical = Chem.MolToSmiles(m.m, canonical=True)

        # Проверяем, есть ли уже
        existing = db.query(CachedName).filter(CachedName.smiles == canonical).first()
        if existing:
            if entry.name:
                existing.name = entry.name
            db.commit()
            return {"status": get_text("lib_updated", lang), "smiles": canonical}

        new_entry = CachedName(smiles=canonical, name=entry.name or "")
        db.add(new_entry)
        db.commit()
        return {"status": get_text("lib_saved", lang), "smiles": canonical}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/library", summary=get_text("ep_library_desc"))
def list_library(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Return the list of saved molecules."""
    try:
        entries = db.query(CachedName).offset(offset).limit(limit).all()
        total = db.query(CachedName).count()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "entries": [{"smiles": e.smiles, "name": e.name} for e in entries],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/library/{smiles}", summary=get_text("ep_library_delete_desc"))
def delete_from_library(
    smiles: str,
    lang: str = Depends(get_lang),
    db: Session = Depends(get_db),
):
    """Delete a molecule from the library by SMILES."""
    try:
        entry = db.query(CachedName).filter(CachedName.smiles == smiles).first()
        if not entry:
            raise HTTPException(status_code=404, detail=get_text("molecule_not_found_in_library", lang))
        db.delete(entry)
        db.commit()
        return {"status": get_text("lib_deleted", lang), "smiles": smiles}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Сравнение нескольких молекул
# ─────────────────────────────────────────────────────────────────────────────


@router.post("/compare", summary=get_text("ep_compare_desc"))
def compare_molecules(
    smiles_list: List[str] = Query(..., description=get_text("param_smiles_list")),
    lang: str = Depends(get_lang),
):
    """
    Compare several molecules pairwise.
    Returns a Tanimoto similarity matrix.
    """
    try:
        mols = []
        valid_smiles = []
        for smiles in smiles_list:
            m = Molecule(smiles)
            if m.m is not None:
                mols.append(m.m)
                valid_smiles.append(smiles)

        if len(mols) < 2:
            raise ValueError(get_text("need_at_least_two_smiles", lang))

        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]

        n = len(mols)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                matrix[i][j] = round(DataStructs.TanimotoSimilarity(fps[i], fps[j]), 4)

        return {"smiles": valid_smiles, "count": n, "similarity_matrix": matrix}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Рендеринг с подсветкой функциональных групп
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/render_highlighted", summary=get_text("ep_render_highlighted_desc"))
def render_highlighted(
    smiles: str = Query(..., description=get_text("param_smiles")),
    highlight_groups: Optional[List[str]] = Query(
        None, description=get_text("param_highlight_groups")
    ),
    lang: str = Depends(get_lang),
):
    """Render a molecule with functional groups highlighted."""
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        mol = m.m
        highlight_atoms = []
        highlight_colors = {}

        if highlight_groups:
            for group_key in highlight_groups:
                if group_key in FUNCTIONAL_GROUPS:
                    pattern = Chem.MolFromSmarts(FUNCTIONAL_GROUPS[group_key]["smarts"])
                    if pattern:
                        matches = mol.GetSubstructMatches(pattern)
                        for match in matches:
                            for atom_idx in match:
                                if atom_idx not in highlight_atoms:
                                    highlight_atoms.append(atom_idx)
                                    highlight_colors[atom_idx] = (
                                        0.2,
                                        0.8,
                                        0.2,
                                    )  # Зелёный

        img = Draw.MolToImage(
            mol,
            size=(500, 500),
            highlightAtoms=highlight_atoms if highlight_atoms else None,
            highlightAtomColors=highlight_colors if highlight_colors else None,
        )

        from io import BytesIO
        import base64

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "smiles": smiles,
            "image_base64": b64_str,
            "highlighted_atoms": highlight_atoms,
            "highlighted_groups": highlight_groups or [],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# NMR (1H and 13C) — heuristic estimate
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/nmr", summary=get_text("ep_nmr_desc"))
def estimate_nmr(
    smiles: str = Query(..., description=get_text("param_smiles")),
    lang: str = Depends(get_lang),
):
    """Heuristic estimate of NMR chemical shifts that takes the chemical environment into account."""
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        mol = Chem.AddHs(m.m)

        # 1H ЯМР: анализ по функциональным центрам и метильным/метиленовым группам
        proton_shifts = []
        
        # 1. Поиск карбоновых кислот (-COOH)
        acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
        if acid_pattern:
            for match in mol.GetSubstructMatches(acid_pattern):
                h_idx = match[2] # Индекс протона у кислорода в базовом патерне или найдем через соседи
                oxy_atom = mol.GetAtomWithIdx(match[2])
                for n in oxy_atom.GetNeighbors():
                    if n.GetAtomicNum() == 1:
                        proton_shifts.append({
                            "group": "COOH",
                            "shift_range": [10.5, 12.5],
                            "description": get_text("nmr_cooh", lang),
                            "multiplicity": "s",
                            "integration": 1
                        })

        # 2. Поиск метильных групп (-CH3)
        methyl_pattern = Chem.MolFromSmarts("[CX4H3]")
        if methyl_pattern:
            for match in mol.GetSubstructMatches(methyl_pattern):
                c_atom = mol.GetAtomWithIdx(match[0])
                # Проверим, примыкает ли к карбонилу (например, ацетаты/уксусная кислота)
                is_near_carbonyl = False
                for n in c_atom.GetNeighbors():
                    if n.GetAtomicNum() == 6:
                        for nn in n.GetNeighbors():
                            if nn.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(n.GetIdx(), nn.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
                                is_near_carbonyl = True
                
                if is_near_carbonyl:
                    shift_range = [2.0, 2.3]
                    desc = get_text("nmr_ch3_near_carbonyl", lang)
                else:
                    shift_range = [0.8, 1.2]
                    desc = get_text("nmr_ch3_aliphatic", lang)

                # Собираем сами атомы водорода для этого метила
                h_count = sum(1 for n in c_atom.GetNeighbors() if n.GetAtomicNum() == 1)
                if h_count > 0:
                    proton_shifts.append({
                        "group": "CH3",
                        "shift_range": shift_range,
                        "description": desc,
                        "multiplicity": "s" if is_near_carbonyl else "t",
                        "integration": h_count
                    })

        # 13C ЯМР
        carbon_shifts = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # Углерод
                c_shift = _estimate_13c_shift_improved(mol, atom, lang)
                if c_shift:
                    carbon_shifts.append({
                        "atom_idx": atom.GetIdx(),
                        "shift_range": c_shift["range"],
                        "description": c_shift["desc"],
                    })

        return {
            "smiles": smiles,
            "proton_nmr": {
                "shifts": proton_shifts,
            },
            "carbon_nmr": {
                "shifts": carbon_shifts,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _estimate_13c_shift_improved(mol, c_atom, lang="en"):
    """Refined chemical-shift estimate for 13C NMR."""
    c_idx = c_atom.GetIdx()
    
    # Проверяем карбонильные группы
    for neighbor in c_atom.GetNeighbors():
        if neighbor.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(c_idx, neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
            # Проверяем, кислота ли это / сложный эфир / кетон
            is_acid_or_ester = False
            for n2 in c_atom.GetNeighbors():
                if n2.GetAtomicNum() == 8 and n2.GetIdx() != neighbor.GetIdx():
                    is_acid_or_ester = True
            if is_acid_or_ester:
                return {"range": [165, 185], "desc": get_text("nmr_carbonyl_acid_ester", lang)}
            return {"range": [190, 220], "desc": get_text("nmr_carbonyl_ketone_aldehyde", lang)}

    # Ароматические углероды
    if c_atom.GetIsAromatic():
        return {"range": [110, 160], "desc": get_text("nmr_aromatic_c", lang)}

    # Алифатические углероды по количеству водородов
    num_h = sum(1 for n in c_atom.GetNeighbors() if n.GetAtomicNum() == 1)
    if num_h >= 3:
        return {"range": [10, 30], "desc": get_text("nmr_methyl", lang)}
    elif num_h == 2:
        return {"range": [15, 45], "desc": get_text("nmr_methylene", lang)}
    else:
        return {"range": [25, 60], "desc": get_text("nmr_ch_methine_quaternary", lang)}


# ─────────────────────────────────────────────────────────────────────────────
# Toxicity (ADMET)
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/toxicity", summary=get_text("ep_toxicity_desc"))
def assess_toxicity(
    smiles: str = Query(..., description=get_text("param_smiles")),
    lang: str = Depends(get_lang),
):
    """
    Assess the potential toxicity of a molecule with the RDKit FilterCatalog:
    - Brenk filters (undesirable fragments for drugs)
    - PAINS (pan-assay interference compounds)
    """
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))
        mol = m.m

        # Подключаем каталоги фильтров (Brenk и PAINS)
        from rdkit.Chem import FilterCatalog
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog.FilterCatalog(params)

        matches = catalog.GetMatches(mol)
        alerts_found = []
        for entry in matches:
            alerts_found.append({
                "name": entry.GetDescription(),
                "heading": entry.GetHeading() if hasattr(entry, 'GetHeading') else get_text("tox_alert", lang)
            })

        if len(alerts_found) >= 2:
            overall_risk = get_text("tox_risk_high", lang)
        elif len(alerts_found) == 1:
            overall_risk = get_text("tox_risk_medium", lang)
        else:
            overall_risk = get_text("tox_risk_low", lang)

        return {
            "smiles": smiles,
            "overall_risk": overall_risk,
            "total_alerts": len(alerts_found),
            "alerts": alerts_found
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Ретросинтез (RDKit RECAP)
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/retrosynthesis", summary=get_text("ep_retrosynthesis_desc"))
def retrosynthetic_analysis(
    smiles: str = Query(..., description=get_text("param_smiles_target")),
    lang: str = Depends(get_lang),
):
    """
    Retrosynthetic analysis of a molecule with the RECAP algorithm
    (Retrosynthetic Combinatorial Analysis Procedure).

    Breaks bonds according to standard rules (amides, esters, ethers,
    ureas, sulfonamides, etc.) and returns the terminal precursor fragments —
    only the leaves of the hierarchical decomposition tree.
    """
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        mol = m.m

        # RecapDecompose возвращает корень иерархического дерева фрагментов
        # (или None, если молекулу не удалось разобрать).
        recap_tree = Recap.RecapDecompose(mol)
        if recap_tree is None:
            return {
                "smiles": smiles,
                "decomposable": False,
                "fragments_count": 0,
                "fragments": [],
            }

        # GetLeaves() — только конечные фрагменты (листья дерева).
        # GetAllChildren() вернул бы ВСЕ узлы, включая промежуточные.
        leaves = recap_tree.GetLeaves()

        fragments = []
        seen = set()
        for leaf_smiles in leaves:
            frag_mol = Chem.MolFromSmiles(leaf_smiles)
            if frag_mol is None:
                continue
            # Канонизируем SMILES для единообразия и дедупликации
            canon_smi = Chem.MolToSmiles(frag_mol)
            if canon_smi in seen:
                continue
            seen.add(canon_smi)
            fragments.append(
                {
                    "smiles": canon_smi,
                    "formula": rdMolDescriptors.CalcMolFormula(frag_mol),
                    "molecular_weight": round(Descriptors.MolWt(frag_mol), 4),
                    "attachment_points": canon_smi.count("*"),
                }
            )

        return {
            "smiles": smiles,
            "decomposable": bool(fragments),
            "fragments_count": len(fragments),
            "fragments": fragments,
            "note": get_text("retro_note", lang),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Конформеры
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/conformers", summary=get_text("ep_conformers_desc"))
def generate_conformers(
    smiles: str = Query(..., description=get_text("param_smiles")),
    num_conformers: int = Query(10, ge=1, le=50, description=get_text("param_num_conformers")),
    lang: str = Depends(get_lang),
):
    """
    Generate several low-energy conformers of a molecule.
    Uses ETKDG for generation and MMFF for optimization.
    """
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        mol = Chem.AddHs(m.m)

        # Генерация конформеров с ETKDG
        params = AllChem.ETKDGv3()
        params.numThreads = 0  # Использовать все доступные потоки
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)

        if len(cids) == 0:
            raise ValueError(get_text("failed_to_generate_conformers", lang))

        # Оптимизация каждого конформера
        results = []
        for cid in cids:
            try:
                energy = _optimize_and_energy(mol, cid)
                results.append({
                    "conf_id": int(cid),
                    "energy": round(energy, 4),
                })
            except Exception:
                continue

        # Сортировка по энергии
        results.sort(key=lambda x: x["energy"])

        # Генерация SDF для каждого конформера
        conformers_sdf = []
        for i, r in enumerate(results):
            conf_mol = Chem.Mol(mol)
            conf_mol.RemoveAllConformers()
            conf_mol.AddConformer(mol.GetConformer(r["conf_id"]))
            sdf_block = Chem.MolToMolBlock(conf_mol)
            conformers_sdf.append({
                "rank": i + 1,
                "conf_id": r["conf_id"],
                "energy": r["energy"],
                "sdf": sdf_block,
            })

        return {
            "smiles": smiles,
            "total_conformers": len(results),
            "conformers": conformers_sdf,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Экспорт в SVG
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/export_svg", summary=get_text("ep_export_svg_desc"))
def export_svg(
    smiles: str = Query(..., description=get_text("param_smiles")),
    width: int = Query(400, ge=100, le=2000, description=get_text("param_width")),
    height: int = Query(400, ge=100, le=2000, description=get_text("param_height")),
    lang: str = Depends(get_lang),
):
    """
    Export the 2D structure of a molecule as SVG.
    Suitable for inclusion in scientific publications.
    """
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        mol = m.m

        # Генерация 2D-координат
        AllChem.Compute2DCoords(mol)

        # Рендеринг в SVG
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        return Response(content=svg, media_type="image/svg+xml")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Масс-спектрометрия (изотопное распределение)
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/ms", summary=get_text("ep_ms_desc"))
def mass_spectrometry(
    smiles: str = Query(..., description=get_text("param_smiles")),
    lang: str = Depends(get_lang),
):
    """Calculate the expected mass spectrum and isotope distribution with pyOpenMS."""
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        mol = m.m
        exact_mass = Descriptors.ExactMolWt(mol)
        formula_str = rdMolDescriptors.CalcMolFormula(mol)

        import pyopenms as oms

        # Создаем формулу напрямую из корректной строки RDKit
        formula = oms.EmpiricalFormula(formula_str)

        # Генератор изотопных паттернов (4 пика)
        generator = oms.CoarseIsotopePatternGenerator(4)
        isotope_distribution = formula.getIsotopeDistribution(generator)

        # Извлекаем пики (переводим интенсивность в проценты для наглядности)
        peaks = []
        for peak in isotope_distribution.getContainer():
            peaks.append({
                "m_z": round(peak.getMZ(), 4),
                "intensity_percent": round(peak.getIntensity() * 100, 4)
            })

        return {
            "smiles": smiles,
            "exact_mass": round(exact_mass, 6),
            "molecular_formula": formula_str,
            "isotope_pattern": peaks,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def _calculate_isotope_pattern(atom_counts, exact_mass):
    """Calculate the isotope distribution with pyOpenMS."""
    try:
        import pyopenms as oms

        # Формируем формулу строкой
        formula_str = ""
        for elem, count in sorted(atom_counts.items()):
            if count > 0:
                formula_str += elem
                if count > 1:
                    formula_str += str(count)

        # Создаём формулу
        formula = oms.EmpiricalFormula(formula_str)

        # Генератор изотопных паттернов (4 пика)
        generator = oms.CoarseIsotopePatternGenerator(4)
        isotope_distribution = formula.getIsotopeDistribution(generator)

        # Извлекаем пики
        peaks = []
        for peak in isotope_distribution.getContainer():
            peaks.append((round(peak.getMZ(), 4), round(peak.getIntensity(), 6)))

        # Сортировка по интенсивности
        peaks.sort(key=lambda x: x[1], reverse=True)

        # Нормализация к максимуму
        if peaks:
            max_intensity = peaks[0][1]
            peaks = [(m, round(i / max_intensity, 4)) for m, i in peaks]

        return peaks

    except ImportError:
        # Fallback на упрощённый расчёт, если pyOpenMS недоступен
        return _calculate_isotope_pattern_fallback(atom_counts, exact_mass)


def _calculate_isotope_pattern_fallback(atom_counts, exact_mass):
    """Simplified isotope-distribution calculation (fallback)."""
    isotopes = {
        "C": [(12.0, 0.989), (13.003, 0.011)],
        "H": [(1.008, 0.9998), (2.014, 0.0002)],
        "N": [(14.003, 0.996), (15.000, 0.004)],
        "O": [(15.995, 0.998), (17.999, 0.002)],
        "S": [(31.972, 0.95), (33.968, 0.04)],
        "Cl": [(34.969, 0.75), (36.966, 0.25)],
        "Br": [(78.918, 0.50), (80.916, 0.50)],
    }

    peaks = [(round(exact_mass, 4), 1.0)]

    for elem, count in atom_counts.items():
        if elem not in isotopes:
            continue
        elem_isotopes = isotopes[elem]
        if len(elem_isotopes) < 2:
            continue

        new_peaks = []
        for mass, intensity in peaks:
            new_peaks.append((mass, intensity * elem_isotopes[0][1] ** count))
            if count >= 1:
                delta_mass = elem_isotopes[1][0] - elem_isotopes[0][0]
                new_peaks.append((
                    round(mass + delta_mass * count, 4),
                    intensity * elem_isotopes[1][1] * count
                ))
        peaks = new_peaks

    peaks.sort(key=lambda x: x[1], reverse=True)

    max_intensity = peaks[0][1] if peaks else 1.0
    normalized = [(m, round(i / max_intensity, 4)) for m, i in peaks[:10]]

    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# Тавтомеры
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/tautomers", summary=get_text("ep_tautomers_desc"))
def enumerate_tautomers(
    smiles: str = Query(..., description=get_text("param_smiles")),
    lang: str = Depends(get_lang),
):
    """
    Enumerate all possible tautomers of a molecule.

    Uses the RDKit TautomerEnumerator to generate every unique tautomeric
    form of the molecule.

    **Parameters:**
    - `smiles` — molecule SMILES string
    - `lang` — response language (en, ru, uk, es)

    **Returns:**
    - `canonical_tautomer` — canonical (most stable) tautomer
    - `tautomers` — list of all unique tautomers as SMILES
    - `count` — number of tautomers found

    **Example request:**
    ```
    GET /rdkit/api/tautomers?smiles=CC(=O)C&lang=en
    ```

    **Example response:**
    ```json
    {
        "smiles": "CC(=O)C",
        "canonical_tautomer": "CC(=O)C",
        "tautomers": ["CC(=O)C", "C=C(O)C"],
        "count": 2
    }
    ```
    """
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        enumerator = rdMolStandardize.TautomerEnumerator()
        canonical = enumerator.Canonicalize(m.m)
        tautomers = enumerator.Enumerate(m.m)

        canonical_smiles = Chem.MolToSmiles(canonical, canonical=True)
        all_tautomers = [Chem.MolToSmiles(t, canonical=True) for t in tautomers]

        return {
            "smiles": smiles,
            "canonical_tautomer": canonical_smiles,
            "tautomers": all_tautomers,
            "count": len(all_tautomers),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Bemis-Murcko Scaffold
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/scaffold", summary=get_text("ep_scaffold_desc"))
def get_scaffold(
    smiles: str = Query(..., description=get_text("param_smiles")),
    lang: str = Depends(get_lang),
):
    """
    Extract the Bemis-Murcko scaffold from a molecule.

    The Bemis-Murcko algorithm removes side chains, leaving only the ring
    skeleton and the bonds connecting the rings.

    **Parameters:**
    - `smiles` — molecule SMILES string
    - `lang` — response language (en, ru, uk, es)

    **Returns:**
    - `scaffold_smiles` — SMILES of the scaffold
    - `scaffold_svg` — SVG image of the scaffold

    **Example request:**
    ```
    GET /rdkit/api/scaffold?smiles=CC(=O)Oc1ccccc1C(=O)O&lang=en
    ```

    **Example response:**
    ```json
    {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "scaffold_smiles": "O=C(O)c1ccccc1",
        "scaffold_svg": "<svg>...</svg>"
    }
    ```
    """
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        scaffold = MurckoScaffold.GetScaffoldForMol(m.m)
        scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)

        # Генерация SVG
        drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
        drawer.DrawMolecule(scaffold)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        return {
            "smiles": smiles,
            "scaffold_smiles": scaffold_smiles,
            "scaffold_svg": svg,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Butina Clustering
# ─────────────────────────────────────────────────────────────────────────────


class ClusterRequest(BaseModel):
    """Request model for molecule clustering."""

    smiles_list: List[str] = Field(..., description=get_text("param_smiles_list_cluster"))
    cutoff: float = Field(0.7, ge=0.0, le=1.0, description=get_text("param_cutoff"))


@router.post("/cluster", summary=get_text("ep_cluster_desc"))
def cluster_molecules(
    request: ClusterRequest,
    lang: str = Depends(get_lang),
):
    """
    Cluster molecules with the Butina algorithm.

    Uses Morgan fingerprints (radius 2, 2048 bits) and a Tanimoto similarity
    matrix to group molecules by structural similarity.

    **Request parameters (JSON):**
    - `smiles_list` — list of SMILES strings (at least 2)
    - `cutoff` — Tanimoto similarity threshold (0.0–1.0, default 0.7)

    **Returns:**
    - `total_molecules` — number of valid molecules
    - `cutoff` — threshold used
    - `cluster_count` — number of clusters
    - `clusters` — list of clusters with SMILES

    **Example request:**
    ```json
    POST /rdkit/api/cluster
    {
        "smiles_list": ["CCO", "CCCO", "c1ccccc1", "c1ccccc1O"],
        "cutoff": 0.7
    }
    ```

    **Example response:**
    ```json
    {
        "total_molecules": 4,
        "cutoff": 0.7,
        "cluster_count": 2,
        "clusters": [
            {"cluster_id": 0, "size": 2, "smiles": ["CCO", "CCCO"]},
            {"cluster_id": 1, "size": 2, "smiles": ["c1ccccc1", "c1ccccc1O"]}
        ]
    }
    ```
    """
    try:
        mols = []
        valid_smiles = []
        for smiles in request.smiles_list:
            m = Molecule(smiles)
            if m.m is not None:
                mols.append(m.m)
                valid_smiles.append(smiles)

        if len(mols) < 2:
            raise ValueError(get_text("need_at_least_two_smiles", lang))

        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]

        # Вычисление матрицы расстояний (1 - Tanimoto)
        dists = []
        for i in range(1, len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1 - x for x in sims])

        clusters = Butina.ClusterData(dists, len(fps), 1.0 - request.cutoff, isDistData=True)  # cutoff — сходство, Butina ждёт расстояние

        result = []
        for cluster in clusters:
            result.append(
                {
                    "cluster_id": len(result),
                    "size": len(cluster),
                    "smiles": [valid_smiles[i] for i in cluster],
                }
            )

        return {
            "total_molecules": len(valid_smiles),
            "cutoff": request.cutoff,
            "cluster_count": len(clusters),
            "clusters": result,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Стереоизомеры
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/stereoisomers", summary=get_text("ep_stereoisomers_desc"))
def enumerate_stereoisomers(
    smiles: str = Query(..., description=get_text("param_smiles")),
    lang: str = Depends(get_lang),
):
    """
    Enumerate all possible stereoisomers of a molecule.

    Uses the RDKit EnumerateStereoisomers algorithm to generate every unique
    stereoisomer with undefined stereochemistry.

    **Parameters:**
    - `smiles` — molecule SMILES string
    - `lang` — response language (en, ru, uk, es)

    **Returns:**
    - `stereoisomers` — list of all stereoisomers as SMILES
    - `count` — number of stereoisomers found
    - `chiral_centers_count` — number of chiral centres (N)
    - `chiral_centers` — list of chiral centres
    - `max_theoretical_stereoisomers` — theoretical maximum (2^N)
    - `formula` — formula string (for example, "2^2 = 4")

    **Formula check:**
    The number of stereoisomers is 2^N, where N is the number of chiral
    centres. For example, a molecule with 2 chiral centres has 2^2 = 4
    stereoisomers.

    **Example request:**
    ```
    GET /rdkit/api/stereoisomers?smiles=CC(O)C&lang=en
    ```

    **Example response:**
    ```json
    {
        "smiles": "CC(O)C",
        "stereoisomers": ["C[C@H](O)C", "C[C@@H](O)C"],
        "count": 2,
        "chiral_centers_count": 1,
        "chiral_centers": [[1, "?"]],
        "max_theoretical_stereoisomers": 2,
        "formula": "2^1 = 2"
    }
    ```
    """
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        # Находим хиральные центры
        chiral_centers = Chem.FindMolChiralCenters(m.m, includeUnassigned=True)
        num_chiral = len(chiral_centers)
        max_theoretical = 2 ** num_chiral if num_chiral > 0 else 1

        opts = StereoEnumerationOptions(tryEmbedding=True, unique=True)
        isomers = list(EnumerateStereoisomers(m.m, options=opts))

        isomer_smiles = [
            Chem.MolToSmiles(iso, canonical=True, isomericSmiles=True) for iso in isomers
        ]

        return {
            "smiles": smiles,
            "stereoisomers": isomer_smiles,
            "count": len(isomer_smiles),
            "chiral_centers_count": num_chiral,
            "chiral_centers": chiral_centers,
            "max_theoretical_stereoisomers": max_theoretical,
            "formula": f"2^{num_chiral} = {max_theoretical}",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 3D-конформеры (интерактивная визуализация)
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/3d_conformers", summary=get_text("ep_3d_conformers_desc"))
def get_3d_conformers(
    smiles: str = Query(..., description=get_text("param_smiles")),
    num_conformers: int = Query(5, ge=1, le=20, description=get_text("param_num_conformers")),
    lang: str = Depends(get_lang),
):
    """
    Generate 3D conformers of a molecule with interactive visualization.

    Returns an HTML page with a 3Dmol.js viewer:
    - rotate the molecule with the mouse
    - show the energy of each conformer
    - switch between conformers
    - automatic spin

    **Parameters:**
    - `smiles` — molecule SMILES string
    - `num_conformers` — number of conformers (1–20, default 5)
    - `lang` — response language (en, ru, uk, es)

    **Returns:**
    - an HTML page with interactive 3D visualization

    **Example request:**
    ```
    GET /rdkit/api/3d_conformers?smiles=CCO&num_conformers=5&lang=en
    ```

    **Note:**
    Open the response in a browser. The page uses 3Dmol.js for interactive
    visualization of the molecule.
    """
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        mol = Chem.AddHs(m.m)

        # Generate conformers
        params = AllChem.ETKDGv3()
        params.numThreads = 0
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)

        if len(cids) == 0:
            raise ValueError(get_text("failed_to_generate_conformers", lang))

        # Optimize and collect data
        conformers = []
        for cid in cids:
            try:
                energy = _optimize_and_energy(mol, cid)
                sdf = Chem.MolToMolBlock(mol, confId=cid)
                conformers.append(
                    {
                        "conf_id": int(cid),
                        "energy": round(energy, 4),
                        "sdf": sdf,
                    }
                )
            except Exception:
                continue

        # Сортировка по энергии
        conformers.sort(key=lambda x: x["energy"])
        if not conformers:
            raise ValueError(get_text("failed_to_generate_conformers", lang))

        conformers_json = json.dumps(conformers).replace("</", "<\\/")
        smiles_html = html_lib.escape(smiles)

        # Build the interactive 3Dmol.js page
        html_content = f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
    <meta charset="utf-8">
    <title>{get_text("conf3d_title", lang)} — {smiles_html}</title>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        #viewer {{ width: 600px; height: 500px; position: relative; }}
        .controls {{ margin: 10px 0; }}
        .info {{ margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 5px; }}
        button {{ padding: 8px 16px; margin: 5px; cursor: pointer; }}
        button.active {{ background: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h2>{get_text("conf3d_title", lang)}: {smiles_html}</h2>
    <div class="info">
        <strong>{get_text("conf3d_conformer", lang)}:</strong> <span id="conf-id">1</span> |
        <strong>{get_text("conf3d_energy", lang)}:</strong> <span id="conf-energy">{conformers[0]["energy"] if conformers else 0}</span> {get_text("conf3d_energy_unit", lang)} |
        <strong>{get_text("conf3d_total", lang)}:</strong> {len(conformers)}
    </div>
    <div class="controls">
        <button onclick="prevConformer()">{get_text("conf3d_prev", lang)}</button>
        <button onclick="nextConformer()">{get_text("conf3d_next", lang)}</button>
        <button onclick="toggleSpin()">{get_text("conf3d_spin", lang)}</button>
        <button onclick="resetView()">{get_text("conf3d_reset", lang)}</button>
    </div>
    <div id="viewer"></div>

    <script>
        const conformers = {conformers_json};
        let currentConf = 0;
        let spinning = false;
        let viewer = null;

        function initViewer() {{
            let element = document.getElementById('viewer');
            let config = {{ backgroundColor: 'white' }};
            viewer = $3Dmol.createViewer(element, config);
            loadConformer(0);
            viewer.zoomTo();
            viewer.render();
        }}

        function loadConformer(idx) {{
            currentConf = idx;
            let conf = conformers[idx];
            viewer.clear();
            viewer.addModel(conf.sdf, 'sdf');
            viewer.setStyle({{}}, {{stick: {{}}}});

            document.getElementById('conf-id').textContent = idx + 1;
            document.getElementById('conf-energy').textContent = conf.energy;

            viewer.zoomTo();
            viewer.render();
        }}

        function nextConformer() {{
            let next = (currentConf + 1) % conformers.length;
            loadConformer(next);
        }}

        function prevConformer() {{
            let prev = (currentConf - 1 + conformers.length) % conformers.length;
            loadConformer(prev);
        }}

        function toggleSpin() {{
            spinning = !spinning;
            viewer.spin(spinning ? 'y' : false);
        }}

        function resetView() {{
            viewer.zoomTo();
            viewer.render();
        }}

        initViewer();
    </script>
</body>
</html>"""

        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Анализ конформаций колец (Ring Puckering)
# ─────────────────────────────────────────────────────────────────────────────


def _cremer_pople(coords):
    """
    Cremer–Pople puckering parameters (1975) from the 3D coordinates of the
    ring atoms in cyclic order.

    Returns Q/q (Å), theta/phi (degrees) and an approximate conformation
    label. For rings larger than six atoms only the amplitude and phase are
    reported; ``theta`` is None.
    """
    R = np.asarray(coords, dtype=float)
    n = len(R)
    if n < 5:
        return None

    R = R - R.mean(axis=0)
    ang = 2.0 * np.pi * np.arange(n) / n
    r1 = (R * np.sin(ang)[:, None]).sum(axis=0)
    r2 = (R * np.cos(ang)[:, None]).sum(axis=0)
    normal = np.cross(r1, r2)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        # A perfectly planar ring has no preferred normal; use SVD.
        _, _, vh = np.linalg.svd(R)
        normal = vh[-1]
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            return None

    z = R @ (normal / norm)
    amplitude = float(np.sqrt(np.sum(z ** 2)))

    a2 = np.sqrt(2.0 / n) * np.sum(z * np.cos(2 * ang))
    b2 = -np.sqrt(2.0 / n) * np.sum(z * np.sin(2 * ang))
    q2 = float(np.hypot(a2, b2))
    phi = float(np.degrees(np.arctan2(b2, a2)) % 360.0) if q2 > 0.02 else 0.0

    if n == 6:
        q3 = float(np.sqrt(1.0 / n) * np.sum(((-1.0) ** np.arange(n)) * z))
        theta = float(np.degrees(np.arctan2(q2, q3)))
        if amplitude < 0.05:
            theta, phi = 0.0, 0.0
            label = "planar"
        elif theta <= 25 or theta >= 155:
            label = "chair"
        elif 65 <= theta <= 115:
            phase = phi % 60.0
            label = "boat" if (phase <= 15 or phase >= 45) else "twist-boat"
        else:
            label = "half-chair"
        return {
            "Q": round(amplitude, 4),
            "theta": round(theta, 2),
            "phi": round(phi, 2),
            "conformation": label,
        }

    if n == 5:
        if amplitude < 0.05:
            phi = 0.0
            label = "planar"
        else:
            phase = phi % 36.0
            label = "envelope" if (phase <= 9 or phase >= 27) else "twist"
        return {
            "q": round(amplitude, 4),
            "theta": None,
            "phi": round(phi, 2),
            "conformation": label,
        }

    if amplitude < 0.05:
        phi = 0.0
        label = "planar"
    else:
        label = "puckered"
    return {
        "Q": round(amplitude, 4),
        "theta": None,
        "phi": round(phi, 2),
        "conformation": label,
    }


def _cluster_ring_conformers(mol, ring_atoms, cids, cutoff=0.5):
    """Cluster ring conformers by heavy-atom RMSD with the Butina algorithm."""
    if len(cids) <= 1:
        return [[int(cid) for cid in cids]]

    atom_map = [(int(index), int(index)) for index in ring_atoms]
    distances = []
    for i in range(1, len(cids)):
        for j in range(i):
            try:
                distance = rdMolAlign.GetBestRMS(
                    mol, mol, int(cids[i]), int(cids[j]), map=atom_map
                )
            except Exception:
                distance = 1.0e3
            distances.append(float(distance))

    clusters = Butina.ClusterData(
        distances, len(cids), cutoff, isDistData=True, reordering=True
    )
    return [[int(cids[index]) for index in cluster] for cluster in clusters]


def _get_ring_conformation_for_conf(mol, ring_atoms, conf_id):
    """
    Determine the ring conformation for one conformer.

    Uses rdMolTransforms.GetDihedralDeg for the torsion angles.

    ring_atoms: atom indices of the ring
    conf_id: conformer ID
    """
    if len(ring_atoms) < 5:
        return None

    conf = mol.GetConformer(conf_id)
    n = len(ring_atoms)

    # Torsion angles via rdMolTransforms
    dihedrals = []
    for i in range(n):
        i1 = ring_atoms[i]
        i2 = ring_atoms[(i + 1) % n]
        i3 = ring_atoms[(i + 2) % n]
        i4 = ring_atoms[(i + 3) % n]
        angle = rdMolTransforms.GetDihedralDeg(conf, i1, i2, i3, i4)
        dihedrals.append(round(float(angle), 2))

    # Cremer–Pople parameters
    coords = []
    for idx in ring_atoms:
        pos = conf.GetAtomPosition(idx)
        coords.append([pos.x, pos.y, pos.z])
    cp_params = _cremer_pople(coords)

    return {
        "conf_id": int(conf_id),
        "dihedral_angles": dihedrals,
        "cremer_pople": cp_params,
    }


@router.get("/ring_conformation", summary=get_text("ep_ring_conformation_desc"))
def analyze_ring_conformation(
    smiles: str = Query(..., description=get_text("param_smiles")),
    lang: str = Depends(get_lang),
):
    """
    Analyze the conformations of every ring in the molecule.

    Determines the ring conformation type (chair, boat, twist-boat, envelope,
    etc.) and computes the Cremer–Pople puckering parameters (theta, phi, Q).

    **Parameters:**
    - `smiles` — molecule SMILES string
    - `lang` — response language (en, ru, uk, es)

    **Returns:**
    - `total_rings` — number of rings
    - `rings` — ring data:
      - `ring_size` — ring size
      - `atom_indices` — ring atom indices
      - `dihedral_angles` — torsion angles
      - `cremer_pople` — Cremer–Pople parameters:
        - `Q` — puckering amplitude (six-membered rings)
        - `q` — puckering amplitude (five-membered rings)
        - `theta` — polar angle (degrees)
        - `phi` — azimuthal angle (degrees)
        - `conformation` — conformation type

    **Example request:**
    ```
    GET /rdkit/api/ring_conformation?smiles=C1CCCCC1&lang=en
    ```

    **Example response:**
    ```json
    {
        "smiles": "C1CCCCC1",
        "total_rings": 1,
        "rings": [{
            "ring_size": 6,
            "atom_indices": [0, 1, 2, 3, 4, 5],
            "dihedral_angles": [60.0, -60.0, 60.0, -60.0, 60.0, -60.0],
            "cremer_pople": {
                "Q": 0.7,
                "theta": 0.0,
                "phi": 0.0,
                "conformation": "chair"
            }
        }]
    }
    ```
    """
    try:
        m = Molecule(smiles)
        if m.m is None:
            raise ValueError(get_text("invalid_smiles", lang))

        mol = Chem.AddHs(m.m)
        return _analyze_all_ring_conformations(mol, smiles, lang)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _analyze_all_ring_conformations(
    mol: Chem.Mol,
    smiles: str,
    lang: str = "en",
) -> dict:
    """
    Generate 20 conformers, optimize them, cluster ring geometries by RMSD
    and report the most and least stable ring conformation.

    Args:
        mol: RDKit molecule with explicit hydrogens
        smiles: original SMILES string
        lang: response language

    Returns:
        dictionary with the analysis results
    """
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()

    if not rings:
        return {
            "smiles": smiles,
            "total_rings": 0,
            "rings": [],
            "message": get_text("no_rings_found", lang),
            "warning": get_text("ring_conf_warning", lang),
        }

    params = AllChem.ETKDGv3()
    params.numThreads = 0
    params.randomSeed = 0xF00D
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=20, params=params)

    if len(cids) == 0:
        raise ValueError(get_text("failed_to_generate_conformers", lang))

    conf_energies = {}
    for cid in cids:
        energy = 0.0
        try:
            energy = _optimize_and_energy(mol, cid)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol, confId=cid)
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                if ff:
                    energy = ff.CalcEnergy()
            except Exception:
                pass
        conf_energies[int(cid)] = round(float(energy), 4)

    rings_data = []
    for ring in rings:
        ring_atoms = list(ring)
        clusters = _cluster_ring_conformers(mol, ring_atoms, list(cids))
        conformations = []

        for cluster in clusters:
            best_cid = min(cluster, key=lambda cid: conf_energies.get(cid, float("inf")))
            conf_data = _get_ring_conformation_for_conf(mol, ring_atoms, best_cid)
            if not conf_data or not conf_data["cremer_pople"]:
                continue
            conf_data["energy"] = conf_energies.get(best_cid, 0.0)
            conf_data["cluster_size"] = len(cluster)
            conformations.append(conf_data)

        if not conformations:
            continue

        conformations.sort(key=lambda item: item["energy"])
        last_index = len(conformations) - 1
        for index, conformation in enumerate(conformations):
            conformation["is_most_stable"] = index == 0
            conformation["is_least_stable"] = index == last_index

        energies = [item["energy"] for item in conformations]
        rings_data.append(
            {
                "ring_size": len(ring_atoms),
                "atom_indices": ring_atoms,
                "conformations": conformations,
                "total_conformations_found": len(conformations),
                "most_stable_conformation": conformations[0]["cremer_pople"]["conformation"],
                "least_stable_conformation": conformations[-1]["cremer_pople"]["conformation"],
                "min_energy": min(energies),
                "max_energy": max(energies),
            }
        )

    return {
        "smiles": smiles,
        "total_rings": len(rings_data),
        "rings": rings_data,
        "warning": get_text("ring_conf_warning", lang),
    }
