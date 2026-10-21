"""
10-year CVD risk models

Models:

1- Framingham general CVD
2- ACC/AHA Pooled Cohort Equations
3- SCORE2 (2021); calibrated for Asia-Pacific using SCORE2-AP (2024).
4- WHO CVD risk charts 2019; calibrated using GBD 2017, according to the authors' original implementation.

"""

import numpy as np
import pandas as pd

COLS = {
    "sex": "DEM_02",          # female = 1, male = 0
    "age": "DEM_03",          # years
    "tot_chol": "LAB_04",     # mg/dL
    "hdl": "LAB_05",          # mg/dL
    "sbp": "EXM_05",          # mmHg
    "bp_treated": "HTN_03",   # 0 or 1
    "smoker": "SMK_02",       # 0 or 1
    "diabetes": "DBT_01",     # 0 or 1
    "bmi": "EXM_08"           # kg/m^2
}


def _num(df, key):
    return pd.to_numeric(df[COLS[key]]).to_numpy()


def _sex(df, male_value=0):
    return np.where(_num(df, "sex") == male_value, "male", "female")


def _coefficients(table, groups):
    return {term: np.array([table[g][term] for g in groups])
            for term in next(iter(table.values()))}


def _chol_mmol(df, key, unit):
    if unit not in {"mg/dl", "mmol/l"}:
        raise ValueError("chol_unit must be 'mg/dL' or 'mmol/L'")
    return _num(df, key) / (38.67 if unit == "mg/dl" else 1)


def _risk(lp, s0, a=0.0, b=1.0):
    # Cox risk, with optional complementary log-log recalibration.
    log_hazard = np.log(-np.log(s0)) + lp
    return -np.expm1(-np.exp(a + b * log_hazard))


# Framingham general CVD; D'Agostino et al., Circulation 2008.
_FRAMINGHAM_COEF = {
    "male":   {"age": 3.06117, "tc": 1.12370, "hdl": -0.93263,
               "sbp_t": 1.99881, "sbp_u": 1.93303, "smoker": 0.65451,
               "diabetes": 0.57367, "mean": 23.9802, "s0": 0.88936},
    "female": {"age": 2.32888, "tc": 1.20904, "hdl": -0.70833,
               "sbp_t": 2.82263, "sbp_u": 2.76157, "smoker": 0.52873,
               "diabetes": 0.69154, "mean": 26.1931, "s0": 0.95012}
}

def framingham_risk(df):
    c = _coefficients(_FRAMINGHAM_COEF, _sex(df))
    sbp_coef = np.where(_num(df, "bp_treated") == 1, c["sbp_t"], c["sbp_u"])
    lp = (c["age"] * np.log(_num(df, "age"))
          + c["tc"] * np.log(_num(df, "tot_chol"))
          + c["hdl"] * np.log(_num(df, "hdl"))
          + sbp_coef * np.log(_num(df, "sbp"))
          + c["smoker"] * _num(df, "smoker")
          + c["diabetes"] * _num(df, "diabetes"))
    return _risk(lp - c["mean"], c["s0"])


# ACC/AHA Pooled Cohort Equations; Goff et al., 2013, Appendix 7.
_PCE_COEF = {
    "female_white": {"ln_age": -29.799, "ln_age2": 4.884, "ln_tc": 13.540,
                     "ln_age_tc": -3.114, "ln_hdl": -13.578, "ln_age_hdl": 3.149,
                     "sbp_t": 2.019, "age_sbp_t": 0.0, "sbp_u": 1.957, "age_sbp_u": 0.0,
                     "smoker": 7.574, "age_smoker": -1.665, "diabetes": 0.661,
                     "mean": -29.18, "s0": 0.9665},
    "female_black": {"ln_age": 17.114, "ln_age2": 0.0, "ln_tc": 0.940,
                     "ln_age_tc": 0.0, "ln_hdl": -18.920, "ln_age_hdl": 4.475,
                     "sbp_t": 29.291, "age_sbp_t": -6.432, "sbp_u": 27.820, "age_sbp_u": -6.087,
                     "smoker": 0.691, "age_smoker": 0.0, "diabetes": 0.874,
                     "mean": 86.61, "s0": 0.9533},
    "male_white":   {"ln_age": 12.344, "ln_age2": 0.0, "ln_tc": 11.853,
                     "ln_age_tc": -2.664, "ln_hdl": -7.990, "ln_age_hdl": 1.769,
                     "sbp_t": 1.797, "age_sbp_t": 0.0, "sbp_u": 1.764, "age_sbp_u": 0.0,
                     "smoker": 7.837, "age_smoker": -1.795, "diabetes": 0.658,
                     "mean": 61.18, "s0": 0.9144},
    "male_black":   {"ln_age": 2.469, "ln_age2": 0.0, "ln_tc": 0.302,
                     "ln_age_tc": 0.0, "ln_hdl": -0.307, "ln_age_hdl": 0.0,
                     "sbp_t": 1.916, "age_sbp_t": 0.0, "sbp_u": 1.809, "age_sbp_u": 0.0,
                     "smoker": 0.549, "age_smoker": 0.0, "diabetes": 0.645,
                     "mean": 19.54, "s0": 0.8954}
}

def pce_ascvd_risk(df, race_col=None, default_race="white"):
    if race_col is None:
        race = default_race
    else:
        race = df[race_col].astype(str).to_numpy()
    c = _coefficients(_PCE_COEF, np.char.add(np.char.add(_sex(df), "_"), race))
    age = np.log(_num(df, "age"))
    tc = np.log(_num(df, "tot_chol"))
    hdl = np.log(_num(df, "hdl"))
    sbp = np.log(_num(df, "sbp"))
    sbp_coef = np.where(_num(df, "bp_treated") == 1,
                        c["sbp_t"] + c["age_sbp_t"] * age,
                        c["sbp_u"] + c["age_sbp_u"] * age)
    lp = (c["ln_age"] * age + c["ln_age2"] * age**2
          + (c["ln_tc"] + c["ln_age_tc"] * age) * tc
          + (c["ln_hdl"] + c["ln_age_hdl"] * age) * hdl
          + sbp_coef * sbp
          + (c["smoker"] + c["age_smoker"] * age) * _num(df, "smoker")
          + c["diabetes"] * _num(df, "diabetes"))
    return _risk(lp - c["mean"], c["s0"])


# SCORE2 (2021); SCORE2-AP (2024), Supplementary Table 6: Iran = high.
_SCORE2_COEF = {
    "male":   {"cage": 0.3742, "smoker": 0.6012, "csbp": 0.2777, "ctchol": 0.1458,
               "chdl": -0.2698, "smoker_age": -0.0755, "csbp_age": -0.0255,
               "ctchol_age": -0.0281, "chdl_age": 0.0426, "s0": 0.9605},
    "female": {"cage": 0.4648, "smoker": 0.7744, "csbp": 0.3131, "ctchol": 0.1002,
               "chdl": -0.2606, "smoker_age": -0.1088, "csbp_age": -0.0277,
               "ctchol_age": -0.0226, "chdl_age": 0.0613, "s0": 0.9776}
}

_SCORE2_RECAL = {
    "europe": {
        "low":       {"male": (-0.5699, 0.7476), "female": (-0.7380, 0.7019)},
        "moderate":  {"male": (-0.1565, 0.8009), "female": (-0.3143, 0.7701)},
        "high":      {"male": ( 0.3207, 0.9360), "female": ( 0.5710, 0.9369)},
        "very_high": {"male": ( 0.5836, 0.8294), "female": ( 0.9412, 0.8329)}
    },
    "asia_pacific": {
        "low":       {"male": (-0.375229965, 0.62020875),  "female": (-0.986572446, 0.536743779)},
        "moderate":  {"male": ( 0.284885676, 0.778128607), "female": ( 0.08278687,  0.718980326)},
        "high":      {"male": ( 0.778231091, 0.844985356), "female": ( 0.611474287, 0.703624072)},
        "very_high": {"male": ( 0.608975204, 0.679014197), "female": ( 0.502751798, 0.555577072)}
    }
}

def score2_risk(
    df,
    region="high", # "low", "moderate", "high", or "very_high"
    calibration="asia_pacific", # or "europe"
    chol_unit="mg/dl"
):
    sex = _sex(df)
    c = _coefficients(_SCORE2_COEF, sex)
    recal = _SCORE2_RECAL[calibration][region]
    a = np.where(sex == "male", recal["male"][0], recal["female"][0])
    b = np.where(sex == "male", recal["male"][1], recal["female"][1])
    age = (_num(df, "age") - 60) / 5
    sbp = (_num(df, "sbp") - 120) / 20
    tc = _chol_mmol(df, "tot_chol", chol_unit) - 6
    hdl = (_chol_mmol(df, "hdl", chol_unit) - 1.3) / 0.5
    lp = (c["cage"] * age
          + (c["smoker"] + c["smoker_age"] * age) * _num(df, "smoker")
          + (c["csbp"] + c["csbp_age"] * age) * sbp
          + (c["ctchol"] + c["ctchol_age"] * age) * tc
          + (c["chdl"] + c["chdl_age"] * age) * hdl)
    return _risk(lp, c["s0"], a, b)


# WHO CVD risk cahrts 2019; Appendix 1.
_WHO_COEF = {
    "lab": {
        "mi": {
            "male": {"age": 0.0719227, "tc": 0.2284944, "sbp": 0.0132183,
                       "diabetes": 0.6410114, "smoker": 0.5638109, "tc_age": -0.0045806,
                       "sbp_age": -0.0001576, "diabetes_age": -0.0124966, "smoker_age": -0.0182545,
                       "s0": 0.9540108},
            "female": {"age": 0.1020713, "tc": 0.2050377, "sbp": 0.015823,
                       "diabetes": 1.070358, "smoker": 1.053223, "tc_age": -0.0051932,
                       "sbp_age": -0.0001378, "diabetes_age": -0.0234174, "smoker_age": -0.0332666,
                       "s0": 0.9894159}
        },
        "stroke": {
            "male": {"age": 0.0986578, "tc": 0.029526, "sbp": 0.0222629,
                       "diabetes": 0.6268712, "smoker": 0.4981217, "tc_age": 0.00142,
                       "sbp_age": -0.0004147, "diabetes_age": -0.026302, "smoker_age": -0.0150561,
                       "s0": 0.9849257},
            "female": {"age": 0.1056632, "tc": 0.0257782, "sbp": 0.0206278,
                       "diabetes": 0.8581998, "smoker": 0.7443627, "tc_age": -0.0021387,
                       "sbp_age": -0.0004897, "diabetes_age": -0.0209826, "smoker_age": -0.0200822,
                       "s0": 0.9890569}
        }
    },
    "nonlab": {
        "mi": {
            "male": {"age": 0.073593, "bmi": 0.0337219, "sbp": 0.0133937,
                       "smoker": 0.5954767, "bmi_age": -0.0010432, "sbp_age": -0.0001837,
                       "smoker_age": -0.0200831, "s0": 0.9544258},
            "female": {"age": 0.1049418, "bmi": 0.0257616, "sbp": 0.016726,
                       "smoker": 1.093132, "bmi_age": -0.0006537, "sbp_age": -0.0001966,
                       "smoker_age": -0.0343739, "s0": 0.9887124}
        },
        "stroke": {
            "male": {"age": 0.097674, "bmi": 0.0159518, "sbp": 0.0227294,
                       "smoker": 0.4999862, "bmi_age": -0.0003516, "sbp_age": -0.0004374,
                       "smoker_age": -0.0153895, "s0": 0.984826},
            "female": {"age": 0.1046105, "bmi": 0.0036406, "sbp": 0.0216741,
                       "smoker": 0.7399405, "bmi_age": -1.29e-05, "sbp_age": -0.0005311,
                       "smoker_age": -0.0203997, "s0": 0.9885706}
        }
    }
}


# https://github.com/skkceu/stata/blob/main/whocvdrisk_t10_calparms_gbd_sens0.dta
_WHO_RECAL = {
    "north_africa_middle_east": {
        "lab": {
            "mi": {
                "male": {"a": 0.2415885478258133, "b": 0.8756304383277893},
                "female": {"a": -0.14507868885993958, "b": 0.5792591571807861}
            },
            "stroke": {
                "male": {"a": -0.3300977051258087, "b": 0.7052914500236511},
                "female": {"a": -0.5591117739677429, "b": 0.5836478471755981}
            }
        },
        "nonlab": {
            "mi": {
                "male": {"a": 0.2788589298725128, "b": 0.953991711139679},
                "female": {"a": -0.11677289754152298, "b": 0.599382758140564}
            },
            "stroke": {
                "male": {"a": -0.4061658978462219, "b": 0.6707606911659241},
                "female": {"a": -0.49590355157852173, "b": 0.5810301899909973}
            }
        }
    }
}

def who_cvd_risk_2019(
    df,
    model="lab", # or "nonlab"
    chol_unit="mg/dl",
    calibration="north_africa_middle_east"
):
    sex = _sex(df)
    age = _num(df, "age") - 60
    x = {"sbp": _num(df, "sbp") - 120, "smoker": _num(df, "smoker")}
    if model == "lab":
        x.update(tc=_chol_mmol(df, "tot_chol", chol_unit) - 6,
                 diabetes=_num(df, "diabetes"))
    else:
        x["bmi"] = _num(df, "bmi") - 25
    risks = {}
    for endpoint in ("mi", "stroke"):
        c = _coefficients(_WHO_COEF[model][endpoint], sex)
        k = _coefficients(_WHO_RECAL[calibration][model][endpoint], sex)
        lp = c["age"] * age + sum((c[key] + c[key + "_age"] * age) * value
                                  for key, value in x.items())
        risks[endpoint] = _risk(lp, c["s0"], k["a"], k["b"])
    return 1 - (1 - risks["mi"]) * (1 - risks["stroke"])
