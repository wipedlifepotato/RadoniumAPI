"""
Inorganic Chemistry API Router — extended functionality for inorganic chemistry.

Formulas, molar masses and element data come from molmass. Oxidation states
for SMILES are computed by RDKit; formulas use a compact universal solver
based on charge balance.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache, reduce
from math import gcd
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query

from i18n import SUPPORTED_LANGUAGES, get_text, get_lang
from molmass import ELEMENTS as _ELEMENTS, Formula as _MolFormula

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

router = APIRouter(prefix="/inorg/api", tags=["Inorganic Chemistry API"])

PROXY_ADDRESS = "127.0.0.1:9050"


def set_proxy(proxy: str):
    global PROXY_ADDRESS
    PROXY_ADDRESS = proxy


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ALKALI_METALS = {"Li", "Na", "K", "Rb", "Cs", "Fr"}
ALKALINE_EARTH = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}

# Elements missing from the molmass table (Z = 110…118).
_SUPERHEAVY = {
    "Ds": (110, 281.0, "[Rn] 5f14 6d9 7s1", "4, 3, 2, 0"),
    "Rg": (111, 281.0, "[Rn] 5f14 6d10 7s1", "3, 1"),
    "Cn": (112, 285.0, "[Rn] 5f14 6d10 7s2", "2, 0"),
    "Nh": (113, 286.0, "[Rn] 5f14 6d10 7s2 7p1", "3, 1"),
    "Fl": (114, 289.0, "[Rn] 5f14 6d10 7s2 7p2", "4, 2, 0"),
    "Mc": (115, 288.0, "[Rn] 5f14 6d10 7s2 7p3", "3, 1"),
    "Lv": (116, 293.0, "[Rn] 5f14 6d10 7s2 7p4", "4, 2"),
    "Ts": (117, 294.0, "[Rn] 5f14 6d10 7s2 7p5", "-1, 1, 3, 5"),
    "Og": (118, 294.0, "[Rn] 5f14 6d10 7s2 7p6", "2, 0"),
}

# Additional oxidation states absent from molmass, with an "unusualness" cost.
_EXTRA_OX = {
    "H": [(-1, 0.5), (0, 5.0)],
    "O": [(-1, 0.5), (Fraction(-1, 2), 1.0), (Fraction(-1, 3), 1.5), (0, 5.0), (1, 5.0), (2, 0.8)],
    "N": [(-2, 0.4), (-1, 0.8), (0, 1.0), (1, 0.9)],
    "C": [(-3, 0.4), (-2, 0.5), (-1, 0.6), (0, 1.0), (1, 0.8), (3, 0.4)],
    "Cl": [(2, 0.6), (4, 0.5), (6, 0.5)],
    "Br": [(2, 0.6), (4, 0.5), (6, 0.5)],
    "I": [(2, 0.6), (4, 0.5), (6, 0.5)],
    "Xe": [(8, 0.8)],
}

# Solubility table: exceptions for salts that are usually insoluble.
_SOLUBILITY_EXCEPTIONS = {
    "Cl": {"Ag": "insoluble", "Pb": "slightly", "Hg2": "insoluble", "Cu": "insoluble", "Tl": "insoluble"},
    "Br": {"Ag": "insoluble", "Pb": "slightly", "Hg2": "insoluble", "Cu": "insoluble", "Tl": "insoluble"},
    "I": {"Ag": "insoluble", "Pb": "insoluble", "Hg2": "insoluble", "Cu": "insoluble", "Tl": "insoluble", "Hg": "insoluble"},
    "SO4": {"Ba": "insoluble", "Pb": "insoluble", "Ca": "slightly", "Sr": "insoluble", "Ag": "slightly", "Hg2": "insoluble"},
}

_LOCALIZED_ORDER = ("en", "ru", "uk", "es")


class InorgError(Exception):
    """Error carrying a translation key and its formatting arguments."""

    def __init__(self, key: str, **kwargs: Any):
        super().__init__(key)
        self.key = key
        self.kwargs = kwargs


def _detail(exc: Exception, lang: str) -> str:
    """Resolve an exception to a localized HTTP error detail."""
    if isinstance(exc, InorgError):
        return get_text(exc.key, lang, **exc.kwargs)
    return str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Formulas and elements
# ─────────────────────────────────────────────────────────────────────────────

_SINGLE_ION = re.compile(r"^([A-Z][a-z]?)(\d*)([+-])$")
_ELEMENT_TOKEN = re.compile(r"(\d+)([spdf])(\d*)")

_NOBLE_CORES = {
    "He": "1s2",
    "Ne": "1s2 2s2 2p6",
    "Ar": "1s2 2s2 2p6 3s2 3p6",
    "Kr": "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6",
    "Xe": "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6",
    "Rn": "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 6s2 4f14 5d10 6p6",
}
_CORE_ELECTRONS = {"He": 2, "Ne": 10, "Ar": 18, "Kr": 36, "Xe": 54, "Rn": 86}


def _normalize_formula(formula: str) -> str:
    """Normalize a formula to the form understood by molmass."""
    text = formula.strip().replace(" ", "").replace("·", ".").replace("*", ".").replace("−", "-")
    match = _SINGLE_ION.match(text)
    if match:
        element, digits, sign = match.groups()
        charge = int(digits or 1)
        text = f"[{element}]{'' if charge == 1 else charge}{sign}"
    return text


@lru_cache(maxsize=512)
def _formula_data(formula: str) -> Dict[str, Any]:
    """Parse a formula with molmass and return composition, charge and mass."""
    text = _normalize_formula(formula)
    try:
        parsed = _MolFormula(text)
    except Exception as exc:
        raise InorgError("inorg_invalid_formula", formula=formula) from exc

    elements = {
        element: atom.count
        for element, atom in parsed.composition().items()
        if element != "e-" and atom.count
    }
    if not elements:
        raise InorgError("inorg_cannot_parse_formula", formula=formula)

    for element in elements:
        _element_data(element)  # verify that the symbol is known

    molar_mass = sum(_element_data(el)["atomic_mass"] * n for el, n in elements.items())
    return {
        "formula": parsed.formula,
        "elements": elements,
        "charge": int(round(parsed.charge)),
        "molar_mass": float(molar_mass),
    }


def _parse_formula(formula: str) -> Dict[str, int]:
    """Elemental composition of a formula: {element: count}."""
    return dict(_formula_data(formula)["elements"])


def _parse_oxistates(text: str) -> List[Tuple[int, float]]:
    """Parse a molmass string such as ``6, 3*, 2, 0`` into (state, cost) pairs."""
    result: List[Tuple[int, float]] = []
    for token in (text or "").split(","):
        token = token.strip()
        if not token:
            continue
        common = token.endswith("*")
        token = token.rstrip("*")
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        result.append((value, 0.0 if common else 0.5))
    return result


def _econf_core(econf: str) -> Optional[str]:
    match = re.match(r"\s*\[([A-Za-z]+)\]", econf or "")
    return match.group(1) if match else None


def _parse_econf_tokens(text: str) -> List[Tuple[int, str, int]]:
    return [
        (int(n), letter, int(count) if count else 1)
        for n, letter, count in _ELEMENT_TOKEN.findall(text or "")
    ]


def _expand_econf(econf: str) -> List[Tuple[int, str, int]]:
    text = econf or ""
    core = _econf_core(text)
    result: List[Tuple[int, str, int]] = []
    if core:
        result.extend(_parse_econf_tokens(_NOBLE_CORES.get(core, "")))
        text = re.sub(r"^\s*\[[A-Za-z]+\]\s*", "", text)
    result.extend(_parse_econf_tokens(text))
    return result


def _format_econf(econf: str) -> str:
    text = (econf or "").strip()
    core = _econf_core(text)
    rest = re.sub(r"^\s*\[[A-Za-z]+\]\s*", "", text)
    tokens = [f"{n}{letter}{count or 1}" for n, letter, count in _ELEMENT_TOKEN.findall(rest)]
    if core:
        return f"[{core}] " + " ".join(tokens) if tokens else f"[{core}]"
    return " ".join(tokens)


def _format_full_econf(econf: str) -> str:
    return " ".join(f"{n}{letter}{count}" for n, letter, count in _expand_econf(econf))


def _count_unpaired(config: List[Tuple[int, str, int]]) -> int:
    capacity = {"s": 1, "p": 3, "d": 5, "f": 7}
    unpaired = 0
    for _, letter, electrons in config:
        orbitals = capacity[letter]
        unpaired += electrons if electrons <= orbitals else 2 * orbitals - electrons
    return unpaired


@lru_cache(maxsize=128)
def _element_data(symbol: str) -> Dict[str, Any]:
    """Element data from molmass, with a small fallback for superheavy elements."""
    symbol = symbol.strip()
    if symbol and symbol[0].islower():
        symbol = symbol[0].upper() + symbol[1:]

    try:
        element = _ELEMENTS[symbol]
    except KeyError:
        element = None

    if element is not None:
        oxidation = _parse_oxistates(element.oxistates) or [(0, 0.0)]
        return {
            "symbol": element.symbol,
            "name": element.name,
            "atomic_number": element.number,
            "atomic_mass": element.mass,
            "electronegativity": element.eleneg or None,
            "group": element.group,
            "period": element.period,
            "block": element.block,
            "electron_configuration": _format_econf(element.eleconfig),
            "full_configuration": _format_full_econf(element.eleconfig),
            "expanded_configuration": _expand_econf(element.eleconfig),
            "noble_gas_core": _econf_core(element.eleconfig),
            "oxidation_states": oxidation,
        }

    fallback = _SUPERHEAVY.get(symbol)
    if not fallback:
        raise InorgError("inorg_unknown_element", element=symbol)

    number, mass, econf, oxistates = fallback
    return {
        "symbol": symbol,
        "name": symbol,
        "atomic_number": number,
        "atomic_mass": mass,
        "electronegativity": None,
        "group": None,
        "period": 7,
        "block": "d",
        "electron_configuration": _format_econf(econf),
        "full_configuration": _format_full_econf(econf),
        "expanded_configuration": _expand_econf(econf),
        "noble_gas_core": _econf_core(econf),
        "oxidation_states": _parse_oxistates(oxistates) or [(0, 0.0)],
    }


def _is_metal(symbol: str) -> bool:
    data = _element_data(symbol)
    return (
        data["block"] in ("d", "f")
        or symbol in ALKALI_METALS
        or symbol in ALKALINE_EARTH
        or symbol in {"Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Po"}
    )


def _is_transition_metal(symbol: str) -> bool:
    return _element_data(symbol)["block"] == "d"


# ─────────────────────────────────────────────────────────────────────────────
# Oxidation states
# ─────────────────────────────────────────────────────────────────────────────

def _num(value: Fraction) -> Any:
    value = Fraction(value)
    return int(value) if value.denominator == 1 else round(float(value), 4)


def _oxidation_candidates(symbol: str) -> List[Tuple[Fraction, float]]:
    """Oxidation-state candidates sorted by how common they are."""
    data = _element_data(symbol)
    candidates = [(Fraction(value), cost) for value, cost in data["oxidation_states"]]
    candidates.extend((Fraction(value), cost) for value, cost in _EXTRA_OX.get(symbol, []))

    best: Dict[Fraction, float] = {}
    for value, cost in candidates:
        if value not in best or cost < best[value]:
            best[value] = cost
    return sorted(best.items(), key=lambda item: (item[1], abs(item[0])))[:12]


def _state_options(
    symbol: str,
    count: int,
    candidates: List[Tuple[Fraction, float]],
    fixed: Optional[Fraction],
    mixing_penalty: float = 0.4,
) -> List[Tuple[Fraction, float, Tuple[Tuple[Fraction, int], ...]]]:
    """Ways to distribute ``count`` atoms of an element over its states."""
    if fixed is not None:
        return [(fixed * count, 0.0, ((fixed, count),))]

    best: Dict[Fraction, Tuple[float, Tuple[Tuple[Fraction, int], ...]]] = {}

    def add(contribution: Fraction, cost: float, assignment: Tuple[Tuple[Fraction, int], ...]) -> None:
        if contribution not in best or cost < best[contribution][0]:
            best[contribution] = (cost, assignment)

    for state, cost in candidates:
        add(state * count, cost * count, ((state, count),))

    if count > 1:
        splits = range(1, count) if count <= 20 else {1, count // 2, count - 1}
        for index, (state1, cost1) in enumerate(candidates):
            for state2, cost2 in candidates[index + 1:]:
                for split in splits:
                    contribution = state1 * split + state2 * (count - split)
                    cost = cost1 * split + cost2 * (count - split) + mixing_penalty
                    assignment = tuple(
                        sorted(
                            ((state1, split), (state2, count - split)),
                            key=lambda item: (-item[1], item[0]),
                        )
                    )
                    add(contribution, cost, assignment)

    return [(charge, value[0], value[1]) for charge, value in best.items()]


def _format_oxidation_result(assignment: Dict[str, Tuple[Tuple[Fraction, int], ...]]) -> Dict[str, Any]:
    oxidation_states: Dict[str, Any] = {}
    detail: Dict[str, List[Dict[str, Any]]] = {}
    mixed: List[str] = []

    for element, states in assignment.items():
        total = sum(count for _, count in states)
        average = sum(state * count for state, count in states) / total
        oxidation_states[element] = _num(average)
        detail[element] = [
            {
                "state": _num(state),
                "count": count,
                "percent": round(count / total * 100, 2),
            }
            for state, count in states
        ]
        if len(states) > 1:
            mixed.append(element)

    return {
        "oxidation_states": oxidation_states,
        "oxidation_states_detail": detail,
        "mixed_valence": mixed,
    }


def _solve_oxidation_states(elements: Dict[str, int], charge: int = 0) -> Dict[str, Any]:
    """
    Universal oxidation-state solver based on composition and charge.

    Handles carbonyls, cyanides, sulfates, nitrogen oxides, peroxides and
    mixed valence without compound-specific branches.
    """
    elements = {element: int(count) for element, count in elements.items() if count}
    if not elements:
        raise InorgError("inorg_empty_formula")
    if len(elements) == 1:
        element, count = next(iter(elements.items()))
        state = Fraction(charge, count)
        return _format_oxidation_result({element: ((state, count),)})

    fixed: Dict[str, Fraction] = {}

    # Binary nitrogen oxides: N2O (+1), NO (+2), N2O3 (+3), NO2 (+4), N2O5 (+5).
    if set(elements) <= {"N", "O"} and "N" in elements and "O" in elements:
        fixed["O"] = Fraction(-2)
        fixed["N"] = Fraction(-2 * elements["O"] + charge, elements["N"])
        return _format_oxidation_result(
            {element: ((fixed[element], count),) for element, count in elements.items()}
        )

    # Sulfates and persulfates: S:O = 1:4.
    if elements.get("S", 0) > 0 and elements.get("O", 0) == 4 * elements["S"]:
        fixed["S"] = Fraction(6)

    # Metal carbonyls: CO is a neutral ligand, C(+2), O(-2).
    if (
        elements.get("C", 0)
        and elements.get("C", 0) == elements.get("O", 0)
        and not elements.get("H", 0)
        and any(_is_transition_metal(element) for element in elements)
    ):
        fixed["C"] = Fraction(2)
        fixed["O"] = Fraction(-2)

    # Metal cyanides: CN is C(+2), N(-3).
    if (
        elements.get("C", 0)
        and elements.get("C", 0) == elements.get("N", 0)
        and not elements.get("H", 0)
        and not elements.get("O", 0)
        and any(_is_transition_metal(element) for element in elements)
    ):
        fixed["C"] = Fraction(2)
        fixed["N"] = Fraction(-3)

    organic = "C" in elements and "H" in elements and not any(_is_metal(el) for el in elements)

    candidates: Dict[str, List[Tuple[Fraction, float]]] = {}
    for element in elements:
        if element in fixed:
            continue
        if organic and element == "C":
            candidates[element] = [
                (Fraction(-3), 0.2),
                (Fraction(-2), 0.3),
                (Fraction(-1), 0.4),
                (Fraction(0), 1.0),
                (Fraction(1), 0.6),
                (Fraction(2), 0.4),
                (Fraction(3), 0.2),
                (Fraction(4), 0.5),
                (Fraction(-4), 0.5),
            ]
        else:
            candidates[element] = _oxidation_candidates(element)

    # Dynamic programming over charge. State: charge -> (cost, assignment).
    dp: Dict[Fraction, Tuple[float, Dict[str, Tuple[Tuple[Fraction, int], ...]]]] = {
        Fraction(0): (0.0, {})
    }
    for element, count in elements.items():
        options = _state_options(element, count, candidates.get(element, []), fixed.get(element))
        new_dp: Dict[Fraction, Tuple[float, Dict[str, Tuple[Tuple[Fraction, int], ...]]]] = {}
        for current_charge, (current_cost, assignment) in dp.items():
            for delta, cost, states in options:
                new_charge = current_charge + delta
                new_cost = current_cost + cost
                if new_charge not in new_dp or new_cost < new_dp[new_charge][0]:
                    new_dp[new_charge] = (new_cost, {**assignment, element: states})
        dp = new_dp
        if len(dp) > 5000:
            dp = dict(sorted(dp.items(), key=lambda item: item[1][0])[:5000])

    target = Fraction(charge)
    if target in dp:
        return _format_oxidation_result(dp[target][1])

    raise InorgError("inorg_cannot_determine_oxidation", elements=", ".join(sorted(elements)))


def _oxidation_from_smiles(smiles: str) -> Dict[str, Any]:
    """Compute oxidation states from molecular structure with RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InorgError("invalid_smiles")

    rdMolDescriptors.CalcOxidationNumbers(mol)
    grouped: Dict[str, List[int]] = {}
    for atom in mol.GetAtoms():
        value = atom.GetPropsAsDict().get("OxidationNumber")
        if value is None:
            continue
        grouped.setdefault(atom.GetSymbol(), []).append(int(value))

    assignment: Dict[str, Tuple[Tuple[Fraction, int], ...]] = {}
    for element, values in grouped.items():
        counts: Dict[int, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        assignment[element] = tuple(
            (Fraction(state), count)
            for state, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )

    result = _format_oxidation_result(assignment)
    result["formula"] = rdMolDescriptors.CalcMolFormula(mol)
    result["smiles"] = smiles
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reactions
# ─────────────────────────────────────────────────────────────────────────────

def _split_coefficient(term: str) -> Tuple[int, str]:
    match = re.match(r"^(\d+)\s*(.+)$", term.strip())
    if match:
        return int(match.group(1)), match.group(2).strip()
    return 1, term.strip()


def _parse_reaction_side(side: str) -> List[Dict[str, Any]]:
    parts = [part.strip() for part in re.split(r"\s+\+\s+", side) if part.strip()]
    if len(parts) <= 1:
        parts = [part.strip() for part in side.split("+") if part.strip()]

    terms = []
    for part in parts:
        coefficient, formula = _split_coefficient(part)
        data = _formula_data(formula)
        terms.append(
            {
                **data,
                "formula": formula,
                "normalized_formula": data["formula"],
                "coeff": coefficient,
            }
        )
    return terms


def _parse_reaction(reaction: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    for arrow in ("->", "→", "="):
        if arrow in reaction:
            left, right = reaction.split(arrow, 1)
            return _parse_reaction_side(left), _parse_reaction_side(right)
    raise InorgError("inorg_reaction_format")


def _balance_reaction(
    left_terms: List[Dict[str, Any]], right_terms: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Balance a reaction via the null space of the composition and charge matrix."""
    count_left = len(left_terms)
    total = count_left + len(right_terms)
    if total < 2:
        return None

    elements = set()
    for term in left_terms + right_terms:
        elements.update(term["elements"])

    rows = []
    for element in sorted(elements):
        rows.append(
            [Fraction(term["elements"].get(element, 0)) for term in left_terms]
            + [Fraction(-term["elements"].get(element, 0)) for term in right_terms]
        )
    rows.append(
        [Fraction(term["charge"]) for term in left_terms]
        + [Fraction(-term["charge"]) for term in right_terms]
    )

    pivots: List[int] = []
    row_index = 0
    for column in range(total):
        pivot = next((i for i in range(row_index, len(rows)) if rows[i][column] != 0), None)
        if pivot is None:
            continue
        rows[row_index], rows[pivot] = rows[pivot], rows[row_index]
        pivot_value = rows[row_index][column]
        rows[row_index] = [value / pivot_value for value in rows[row_index]]
        for i in range(len(rows)):
            if i != row_index and rows[i][column] != 0:
                factor = rows[i][column]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[row_index])]
        pivots.append(column)
        row_index += 1
        if row_index == len(rows):
            break

    free = [column for column in range(total) if column not in pivots]
    if len(free) != 1:
        return None

    solution = [Fraction(0)] * total
    solution[free[0]] = Fraction(1)
    for index, column in enumerate(pivots):
        solution[column] = -rows[index][free[0]]
    if any(value <= 0 for value in solution):
        return None

    multiplier = reduce(lambda a, b: a * b // gcd(a, b), (value.denominator for value in solution))
    coefficients = [int(value * multiplier) for value in solution]
    common = reduce(gcd, coefficients)
    coefficients = [value // common for value in coefficients]

    def render(coefficient: int, formula: str) -> str:
        return formula if coefficient == 1 else f"{coefficient}{formula}"

    left_text = " + ".join(
        render(coefficients[i], term["formula"]) for i, term in enumerate(left_terms)
    )
    right_text = " + ".join(
        render(coefficients[count_left + i], term["formula"]) for i, term in enumerate(right_terms)
    )
    return {
        "equation": f"{left_text} -> {right_text}",
        "coefficients": coefficients,
    }


def _reaction_agents(
    left_terms: List[Dict[str, Any]],
    right_terms: List[Dict[str, Any]],
    coefficients: List[int],
) -> Dict[str, Any]:
    """Determine the oxidizing/reducing agent from oxidation-state changes."""
    count_left = len(left_terms)
    for index, term in enumerate(left_terms + right_terms):
        term["coeff"] = coefficients[index]

    states: Dict[int, Dict[str, Any]] = {}
    for index, term in enumerate(left_terms + right_terms):
        try:
            result = _solve_oxidation_states(term["elements"], term["charge"])
            states[index] = result["oxidation_states"]
        except InorgError:
            states[index] = {}

    elements = set()
    for term in left_terms + right_terms:
        elements.update(term["elements"])

    transfers = []
    oxidizing = []
    reducing = []

    for element in sorted(elements):
        left_total = 0.0
        left_atoms = 0
        right_total = 0.0
        right_atoms = 0
        for index, term in enumerate(left_terms):
            amount = term["elements"].get(element, 0)
            if amount and element in states.get(index, {}):
                left_total += term["coeff"] * amount * float(states[index][element])
                left_atoms += term["coeff"] * amount
        for index, term in enumerate(right_terms):
            global_index = count_left + index
            amount = term["elements"].get(element, 0)
            if amount and element in states.get(global_index, {}):
                right_total += term["coeff"] * amount * float(states[global_index][element])
                right_atoms += term["coeff"] * amount

        if not left_atoms or not right_atoms:
            continue

        delta = right_total - left_total
        if abs(delta) < 1e-6:
            continue

        direction = "oxidized" if delta > 0 else "reduced"
        transfers.append(
            {
                "element": element,
                "direction": direction,
                "from_state": round(left_total / left_atoms, 3),
                "to_state": round(right_total / right_atoms, 3),
                "electrons": round(abs(delta), 4),
            }
        )

        targets = oxidizing if delta < 0 else reducing
        for term in left_terms:
            if element in term["elements"] and term["formula"] not in targets:
                targets.append(term["formula"])

    return {
        "oxidizing_agent": ", ".join(oxidizing) or None,
        "reducing_agent": ", ".join(reducing) or None,
        "electron_transfer": transfers,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/", include_in_schema=False)
async def inorg_index(lang: str = Depends(get_lang)):
    """Inorganic Chemistry API index page."""
    return {
        "endpoints": {
            "GET /inorg/api/parse_formula": get_text("inorg_ep_parse_formula", lang),
            "GET /inorg/api/moles": get_text("inorg_ep_moles", lang),
            "GET /inorg/api/electronegativity": get_text("inorg_ep_electronegativity", lang),
            "GET /inorg/api/oxidation_states": get_text("inorg_ep_oxidation_states", lang),
            "GET /inorg/api/solubility": get_text("inorg_ep_solubility", lang),
            "GET /inorg/api/redox": get_text("inorg_ep_redox", lang),
            "GET /inorg/api/stoichiometry": get_text("inorg_ep_stoichiometry", lang),
            "GET /inorg/api/orbitals": get_text("inorg_ep_orbitals", lang),
            "GET /inorg/api/element_info": get_text("inorg_ep_element_info", lang),
        }
    }


@router.get(
    "/parse_formula",
    summary=get_text("inorg_ep_parse_formula"),
    description=get_text("inorg_desc_parse_formula"),
)
def parse_formula(
    formula: str = Query(..., description=get_text("inorg_param_formula")),
    lang: str = Depends(get_lang),
):
    """
    Parse a chemical formula and return elements, charge, molar mass and the
    mass fractions of the elements. Hydrates (CuSO4·5H2O) and complex ions
    ([Fe(CN)6]4-) are supported.
    """
    try:
        data = _formula_data(formula)
        elements = data["elements"]
        total_mass = data["molar_mass"]

        element_details = []
        for element, count in elements.items():
            atomic_mass = _element_data(element)["atomic_mass"]
            element_mass = atomic_mass * count
            element_details.append(
                {
                    "element": element,
                    "count": count,
                    "atomic_mass": round(atomic_mass, 6),
                    "total_mass": round(element_mass, 6),
                    "mass_percent": round(element_mass / total_mass * 100, 2) if total_mass else 0.0,
                }
            )

        return {
            "formula": formula,
            "normalized_formula": data["formula"],
            "charge": data["charge"],
            "elements": element_details,
            "molecular_mass": round(total_mass, 4),
            "total_atoms": sum(elements.values()),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_detail(exc, lang)) from exc


@router.get(
    "/moles",
    summary=get_text("inorg_ep_moles"),
    description=get_text("inorg_desc_moles"),
)
def calculate_moles(
    formula: str = Query(..., description=get_text("inorg_param_formula_short")),
    mass: float = Query(..., description=get_text("inorg_param_mass"), gt=0),
    lang: str = Depends(get_lang),
):
    """Calculate moles, number of molecules and atoms from a mass."""
    try:
        data = _formula_data(formula)
        molar_mass = data["molar_mass"]
        if molar_mass <= 0:
            raise InorgError("inorg_molar_mass_failed")

        moles = mass / molar_mass
        return {
            "formula": formula,
            "mass_g": mass,
            "molecular_mass": round(molar_mass, 4),
            "moles": round(moles, 6),
            "molecules": moles * 6.02214076e23,
            "atoms": sum(data["elements"].values()),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_detail(exc, lang)) from exc


@router.get(
    "/electronegativity",
    summary=get_text("inorg_ep_electronegativity"),
    description=get_text("inorg_desc_electronegativity"),
)
def get_electronegativity(
    element1: str = Query(..., description=get_text("inorg_param_element1")),
    element2: str = Query(..., description=get_text("inorg_param_element2")),
    lang: str = Depends(get_lang),
):
    """
    Determine the bond type from the Pauling electronegativity difference:
    ΔEN < 0.5 — nonpolar covalent, 0.5–1.7 — polar covalent, ≥ 1.7 — ionic.
    """
    try:
        en1 = _element_data(element1)["electronegativity"]
        en2 = _element_data(element2)["electronegativity"]
        if en1 is None or en2 is None:
            raise InorgError("inorg_electronegativity_missing")

        delta = abs(en1 - en2)
        if delta < 0.5:
            bond_type = "nonpolar_covalent"
        elif delta < 1.7:
            bond_type = "polar_covalent"
        else:
            bond_type = "ionic"

        return {
            "element1": element1,
            "element2": element2,
            "electronegativity1": en1,
            "electronegativity2": en2,
            "delta": round(delta, 2),
            "bond_type": bond_type,
            "bond_type_text": get_text(f"bond_{bond_type}", lang),
            **{
                f"bond_type_{code}": get_text(f"bond_{bond_type}", code)
                for code in _LOCALIZED_ORDER
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_detail(exc, lang)) from exc


@router.get(
    "/oxidation_states",
    summary=get_text("inorg_ep_oxidation_states"),
    description=get_text("inorg_desc_oxidation_states"),
)
def get_oxidation_states(
    formula: Optional[str] = Query(None, description=get_text("inorg_param_formula_example")),
    smiles: Optional[str] = Query(None, description=get_text("inorg_param_smiles")),
    lang: str = Depends(get_lang),
):
    """
    Determine the oxidation states of the elements.

    When ``smiles`` is provided, RDKit computes the states from molecular
    structure. For a formula, a universal charge-balance solver accounts for
    carbonyls, cyanides, sulfates, peroxides and mixed valence.
    """
    try:
        if smiles:
            return _oxidation_from_smiles(smiles)
        if not formula:
            raise InorgError("inorg_specify_formula_or_smiles")

        data = _formula_data(formula)
        result = _solve_oxidation_states(data["elements"], data["charge"])
        result.update(
            {
                "formula": formula,
                "normalized_formula": data["formula"],
                "charge": data["charge"],
                "elements": data["elements"],
            }
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_detail(exc, lang)) from exc


def _solubility_response(
    cation_name: str,
    anion_name: str,
    solubility: str,
    rule_key: str,
    lang: str,
    **rule_kwargs: Any,
) -> Dict[str, Any]:
    """Build a localized solubility response for all supported languages."""
    text_key = f"solubility_{solubility}"
    response: Dict[str, Any] = {
        "compound": f"{cation_name}{anion_name}",
        "solubility": solubility,
        "solubility_text": get_text(text_key, lang),
        "rule": get_text(rule_key, lang, **rule_kwargs),
        "rule_key": rule_key,
    }
    for code in _LOCALIZED_ORDER:
        response[f"solubility_{code}"] = get_text(text_key, code)
        response[f"rule_{code}"] = get_text(rule_key, code, **rule_kwargs)
    return response


@router.get(
    "/solubility",
    summary=get_text("inorg_ep_solubility"),
    description=get_text("inorg_desc_solubility"),
)
def check_solubility(
    cation: str = Query(..., description=get_text("inorg_param_cation")),
    anion: str = Query(..., description=get_text("inorg_param_anion")),
    lang: str = Depends(get_lang),
):
    """
    Check compound solubility against the solubility rules.
    Returns soluble, insoluble, slightly or unknown.
    """
    try:
        if anion in ("NO3", "CH3COO", "ClO3", "ClO4"):
            return _solubility_response(
                cation, anion, "soluble", "solubility_rule_nitrates", lang
            )

        if anion in ("Cl", "Br", "I"):
            exceptions = _SOLUBILITY_EXCEPTIONS.get(anion, {})
            solubility = exceptions.get(cation, "soluble")
            rule_key = (
                "solubility_rule_exception"
                if cation in exceptions
                else "solubility_rule_halides"
            )
            return _solubility_response(
                cation, anion, solubility, rule_key, lang, cation=cation
            )

        if anion == "SO4":
            exceptions = _SOLUBILITY_EXCEPTIONS.get("SO4", {})
            solubility = exceptions.get(cation, "soluble")
            rule_key = (
                "solubility_rule_exception"
                if cation in exceptions
                else "solubility_rule_sulfates"
            )
            return _solubility_response(
                cation, anion, solubility, rule_key, lang, cation=cation
            )

        if anion == "S":
            if cation in ALKALI_METALS or cation == "NH4":
                solubility = "soluble"
            elif cation in ALKALINE_EARTH:
                solubility = "slightly"
            else:
                solubility = "insoluble"
            return _solubility_response(
                cation, anion, solubility, "solubility_rule_sulfides", lang
            )

        if anion in ("CO3", "PO4", "C2O4"):
            if cation in ALKALI_METALS or cation == "NH4":
                solubility = "soluble"
            elif cation == "Mg":
                solubility = "slightly"
            else:
                solubility = "insoluble"
            return _solubility_response(
                cation, anion, solubility, "solubility_rule_carbonates", lang
            )

        if anion == "OH":
            if cation in ALKALI_METALS or cation == "NH4" or cation in ("Ba", "Sr"):
                solubility = "soluble"
            elif cation == "Ca":
                solubility = "slightly"
            else:
                solubility = "insoluble"
            return _solubility_response(
                cation, anion, solubility, "solubility_rule_hydroxides", lang
            )

        return _solubility_response(
            cation, anion, "unknown", "solubility_rule_unknown", lang
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_detail(exc, lang)) from exc


@router.get(
    "/redox",
    summary=get_text("inorg_ep_redox"),
    description=get_text("inorg_desc_redox"),
)
def analyze_redox(
    reaction: str = Query(..., description=get_text("inorg_param_reaction")),
    lang: str = Depends(get_lang),
):
    """Balance a redox reaction and identify the oxidizing/reducing agents."""
    try:
        left_terms, right_terms = _parse_reaction(reaction)
        balanced = _balance_reaction(left_terms, right_terms)
        if not balanced:
            raise InorgError("inorg_balance_failed")

        agents = _reaction_agents(left_terms, right_terms, balanced["coefficients"])
        return {
            "reaction": reaction,
            "balanced": balanced["equation"],
            "equation": balanced["equation"],
            "coefficients": balanced["coefficients"],
            **agents,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_detail(exc, lang)) from exc


@router.get(
    "/stoichiometry",
    summary=get_text("inorg_ep_stoichiometry"),
    description=get_text("inorg_desc_stoichiometry"),
)
def calculate_stoichiometry(
    reaction: str = Query(..., description=get_text("inorg_param_stoich_reaction")),
    given_mass: Optional[float] = Query(None, description=get_text("inorg_param_given_mass")),
    given_moles: Optional[float] = Query(None, description=get_text("inorg_param_given_moles")),
    given_formula: Optional[str] = Query(None, description=get_text("inorg_param_given_formula")),
    lang: str = Depends(get_lang),
):
    """
    Calculate reaction stoichiometry. If coefficients are not provided, the
    equation is balanced automatically. You may specify the reagent for which
    the mass or amount is given.
    """
    try:
        left_terms, right_terms = _parse_reaction(reaction)
        balanced = None
        if all(term["coeff"] == 1 for term in left_terms + right_terms):
            balanced = _balance_reaction(left_terms, right_terms)
            if balanced:
                for index, term in enumerate(left_terms + right_terms):
                    term["coeff"] = balanced["coefficients"][index]

        if given_mass is not None or given_moles is not None:
            reactant = None
            if given_formula:
                normalized = _formula_data(given_formula)["formula"]
                reactant = next(
                    (
                        term
                        for term in left_terms
                        if term.get("normalized_formula", term["formula"]) == normalized
                    ),
                    None,
                )
            if reactant is None:
                reactant = left_terms[0]

            if given_moles is None:
                given_moles = given_mass / reactant["molar_mass"]

            ratio = given_moles / reactant["coeff"]
            reactants = [
                {
                    "formula": term["formula"],
                    "coeff": term["coeff"],
                    "moles": round(ratio * term["coeff"], 6),
                    "mass_g": round(ratio * term["coeff"] * term["molar_mass"], 4),
                }
                for term in left_terms
            ]
            products = [
                {
                    "formula": term["formula"],
                    "coeff": term["coeff"],
                    "moles": round(ratio * term["coeff"], 6),
                    "mass_g": round(ratio * term["coeff"] * term["molar_mass"], 4),
                }
                for term in right_terms
            ]
            return {
                "reaction": reaction,
                "balanced": balanced["equation"] if balanced else reaction,
                "given": {
                    "reactant": reactant["formula"],
                    "mass_g": given_mass,
                    "moles": round(given_moles, 6),
                },
                "limiting_reagent": reactant["formula"],
                "reactants": reactants,
                "products": products,
            }

        return {
            "reaction": reaction,
            "balanced": balanced["equation"] if balanced else reaction,
            "reactants": [
                {
                    "formula": term["formula"],
                    "coeff": term["coeff"],
                    "molar_mass": round(term["molar_mass"], 4),
                }
                for term in left_terms
            ],
            "products": [
                {
                    "formula": term["formula"],
                    "coeff": term["coeff"],
                    "molar_mass": round(term["molar_mass"], 4),
                }
                for term in right_terms
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_detail(exc, lang)) from exc


@router.get(
    "/orbitals",
    summary=get_text("inorg_ep_orbitals"),
    description=get_text("inorg_desc_orbitals"),
)
def get_orbitals(
    element: str = Query(..., description=get_text("inorg_param_element")),
    lang: str = Depends(get_lang),
):
    """Return electron configuration, orbitals, unpaired and valence electrons."""
    try:
        data = _element_data(element)
        config = data["expanded_configuration"]
        valence = (
            sum(electrons for _, _, electrons in config)
            - _CORE_ELECTRONS.get(data.get("noble_gas_core"), 0)
        )
        return {
            "element": data["symbol"],
            "atomic_number": data["atomic_number"],
            "electron_configuration": data["electron_configuration"],
            "full_configuration": data["full_configuration"],
            "unpaired_electrons": _count_unpaired(config),
            "valence_electrons": valence,
            "orbitals": [
                {"orbital": f"{n}{letter}", "electrons": electrons}
                for n, letter, electrons in config
            ],
            "period": data["period"],
            "group": data["group"],
            "block": data["block"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_detail(exc, lang)) from exc


@router.get(
    "/element_info",
    summary=get_text("inorg_ep_element_info"),
    description=get_text("inorg_desc_element_info"),
)
def get_element_info(
    symbol: str = Query(..., description=get_text("inorg_param_symbol")),
    lang: str = Depends(get_lang),
):
    """Return atomic number, mass, electronegativity, group, period and oxidation states."""
    try:
        data = _element_data(symbol)
        return {
            "symbol": data["symbol"],
            "name": data["name"],
            "atomic_number": data["atomic_number"],
            "atomic_mass": data["atomic_mass"],
            "electronegativity": data["electronegativity"],
            "oxidation_states": sorted({value for value, _ in data["oxidation_states"]}),
            "group": data["group"],
            "period": data["period"],
            "block": data["block"],
            "electron_configuration": data["electron_configuration"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_detail(exc, lang)) from exc
