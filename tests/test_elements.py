"""
Highest-priority test module: the element parser is where the old app's
real, silent data-loss bugs lived (see elements.py docstring).
"""
import elements as els


class TestNormalizeElementSymbol:
    def test_plain_symbol(self):
        assert els.normalize_element_symbol("Cs") == "Cs"
        assert els.normalize_element_symbol("cs") == "Cs"
        assert els.normalize_element_symbol("U") == "U"

    def test_two_letter_symbol_not_redirected(self):
        # "Li" (lithium) must resolve normally, only a bare "L" is the typo alias.
        assert els.normalize_element_symbol("Li") == "Li"

    def test_bare_l_redirects_to_iodine(self):
        assert els.normalize_element_symbol("L") == "I"
        assert els.normalize_element_symbol("l") == "I"

    def test_iodine_name_aliases(self):
        for q in ["iodine", "IODINE", "iodide", "iodate", "periodate", "129I", "I129", "I-129", "129-I"]:
            assert els.normalize_element_symbol(q) == "I"

    def test_unknown_returns_none(self):
        assert els.normalize_element_symbol("Zz") is None
        assert els.normalize_element_symbol("") is None


class TestIsotopeNotation:
    def test_leading_mass_number(self):
        assert els.parse_analyte_elements("137Cs") == ["Cs"]
        assert els.parse_analyte_elements("90Sr") == ["Sr"]
        assert els.parse_analyte_elements("79Se") == ["Se"]
        assert els.parse_analyte_elements("99Tc") == ["Tc"]
        assert els.parse_analyte_elements("129I") == ["I"]

    def test_metastable_isotope(self):
        assert els.parse_analyte_elements("113mCd") == ["Cd"]

    def test_trailing_mass_number(self):
        assert els.parse_analyte_elements("Cs137") == ["Cs"]
        assert els.parse_analyte_elements("Cs-137") == ["Cs"]

    def test_case_tolerant(self):
        assert els.parse_analyte_elements("129i") == ["I"]


class TestCombinedIsotopeBugFix:
    """The headline fix: combined-isotope slash notation used to fall
    through to no element, silently dropping real Pu/Cm mass."""

    def test_pu_combined_isotope(self):
        assert els.parse_analyte_elements("239/240Pu") == ["Pu"]

    def test_cm_combined_isotope(self):
        assert els.parse_analyte_elements("243/244Cm") == ["Cm"]

    def test_combined_isotope_with_metastable(self):
        assert els.parse_analyte_elements("242/243mAm") == ["Am"]

    def test_combined_isotope_case_tolerant(self):
        assert els.parse_analyte_elements("239/240pu") == ["Pu"]


class TestKnownAliases:
    def test_utotal_variants(self):
        for q in ["UTOTAL", "U_TOTAL", "TOTALU", "TOTAL_U"]:
            assert els.parse_analyte_elements(q) == ["U"]

    def test_carbon_aliases(self):
        assert els.parse_analyte_elements("TOC") == ["C"]
        assert els.parse_analyte_elements("TIC") == ["C"]

    def test_tic_as_co3_prefix_alias(self):
        # Must resolve via the TICAS prefix alias to C/O, NOT to Iodine.
        assert els.parse_analyte_elements("TIC as CO3") == ["C", "O"]

    def test_multi_element_group_analytes(self):
        assert els.parse_analyte_elements("NO3") == ["N", "O"]
        assert els.parse_analyte_elements("SO4") == ["S", "O"]
        assert els.parse_analyte_elements("PO4") == ["P", "O"]

    def test_free_oh_alias(self):
        assert els.parse_analyte_elements("Free OH") == ["O", "H"]

    def test_technetium_exact_tc_only(self):
        assert els.parse_analyte_elements("TC") == ["Tc"]


class TestChemistryNameAliases:
    def test_acetate_is_not_actinium(self):
        assert els.parse_analyte_elements("Acetate") == ["C", "H", "O"]

    def test_nitrate_and_sulfate(self):
        assert els.parse_analyte_elements("Nitrate") == ["N", "O"]
        assert els.parse_analyte_elements("Sulfate") == ["S", "O"]

    def test_organic_solvent_compounds(self):
        assert els.parse_analyte_elements("Carbon Tetrachloride") == ["C", "Cl"]
        assert els.parse_analyte_elements("Vinyl Chloride") == ["C", "H", "Cl"]


class TestFalsePositiveRejection:
    """Regression coverage for the parser's conservative design: it must
    NOT mistake capitalized substrings in natural-language labels for
    element symbols."""

    def test_total_alpha_is_not_aluminum(self):
        assert els.parse_analyte_elements("TotalAlpha") == []
        assert els.parse_analyte_elements("Total Alpha") == []

    def test_total_beta_gamma(self):
        assert els.parse_analyte_elements("Total Beta") == []
        assert els.parse_analyte_elements("Total Gamma") == []

    def test_aroclors_is_not_argon(self):
        assert els.parse_analyte_elements("Aroclors (Total PCB)") == []

    def test_organic_hazardous_analytes_return_empty(self):
        # Real Hanford.csv analyte labels (EPA/RCRA organics) that are
        # genuinely not elemental/radionuclide analytes.
        for label in ["Benzene", "Chloroform", "Toluene", "Xylenes (total)",
                      "N-Nitrosodimethylamine"]:
            assert els.parse_analyte_elements(label) == []


class TestParseAnalyteElementsEdgeCases:
    def test_none_input(self):
        assert els.parse_analyte_elements(None) == []

    def test_whitespace_only_input(self):
        assert els.parse_analyte_elements("   ") == []

    def test_prefix_alias_not_shadowed_by_exact_alias(self):
        # "TIC-190" starts with the TIC prefix but is not itself an exact
        # ANALYTE_ALIASES key (unlike "TIC as CO3" / "TICASCO3"), so this is
        # the only way to actually exercise the ANALYTE_PREFIX_ALIASES loop.
        assert els.parse_analyte_elements("TIC-190") == ["C"]

    def test_aroclor_prefix_without_exact_match(self):
        assert els.parse_analyte_elements("Aroclor 1260") == []

    def test_formula_like_fallback_token_scan(self):
        # Not in any alias table, not isotope notation, not a bare 1-2
        # letter symbol -- must fall through to the final capital-letter
        # token scan (calcium carbonate: Ca, C, O).
        assert els.parse_analyte_elements("CaCO3") == ["Ca", "C", "O"]


class TestTankFarmFromIdFallback:
    def test_fallback_path_used_when_no_leading_digits(self):
        # No leading digit group, so the primary regex can't match; the
        # split-based fallback still finds an alphabetic second token.
        assert els.tank_farm_from_id("Tank-A-1") == "A"

    def test_fallback_returns_none_when_unparseable(self):
        assert els.tank_farm_from_id("12345") is None


class TestClassifyAnalyteEdgeCases:
    def test_none_input(self):
        assert els.classify_analyte(None) == "unknown"

    def test_empty_string(self):
        assert els.classify_analyte("") == "unknown"


class TestElementListHelpers:
    def test_primary_element_from_analyte(self):
        assert els.primary_element_from_analyte("137Cs") == "Cs"
        assert els.primary_element_from_analyte("Benzene") is None

    def test_element_list_string(self):
        assert els.element_list_string("SO4") == "S;O"
        assert els.element_list_string("Benzene") == ""

    def test_element_list_padded(self):
        assert els.element_list_padded("SO4") == ";S;O;"
        assert els.element_list_padded("Benzene") == ";"


class TestTankFarmFromId:
    def test_single_letter_farm(self):
        assert els.tank_farm_from_id("241-A-101") == "A"

    def test_two_letter_farm(self):
        assert els.tank_farm_from_id("241-AN-104") == "AN"

    def test_none_input(self):
        assert els.tank_farm_from_id(None) is None


class TestClassifyAnalyte:
    def test_radionuclide(self):
        assert els.classify_analyte("137Cs") == "radionuclide"

    def test_plain_element(self):
        assert els.classify_analyte("Fe") == "element"

    def test_compound_formula(self):
        assert els.classify_analyte("SO4") == "compound/formula"

    def test_unparsed(self):
        assert els.classify_analyte("Benzene") == "unparsed"
