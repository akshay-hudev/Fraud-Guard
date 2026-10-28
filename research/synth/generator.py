"""Regime-controllable synthetic healthcare-claims generator.

Two fraud typologies are injected into a background of legitimate claims:

* Isolated fraud: single upcoded claims filed through a patient's normal
  doctors. Their signal is in the claim's own features (shift ``isolated_shift``).
* Provider rings: ``ring_doctors`` colluding doctors bill a pool of
  ``ring_patients`` recruited patients. Two independent knobs control them:
    - ``ring_shift`` (delta): feature shift of ring claims, in standard-deviation
      units of log-amount. At 0 a ring claim is drawn from exactly the same
      distribution as a legitimate claim of the same doctor's specialty.
    - ``ring_density`` (s): probability that a pool patient is linked to each
      ring doctor. Ring claim *volume* is fixed, so s changes only how the
      claims are spread over the doctor-patient block (structural signal).
  ``ring_shape`` sets how ring patients are linked to ring doctors:
    - ``block`` (default): each patient-doctor link exists with probability s;
    - ``hub``: one hub doctor is linked to every ring patient and each patient
      to one of the other (spoke) doctors, as in a kickback scheme centred on
      one provider; s is not used;
    - ``pairs``: each patient is linked to exactly two ring doctors drawn at
      random, a distributed ring whose doctor pairs share few patients; s is
      not used.

Time is divided into equal audit periods of ``period_months`` (default four
periods of six months). Periods 1-2 form the training split, period 3 the
validation split and period 4 the test split; every period is analysed on its
own, so count-based features have the same meaning in every split.

Ring timing follows one of two protocols:
* ``emergent`` (default): each ring is active during one period, and rings in
  different periods are formed by disjoint doctors and patients. Test rings
  are therefore new; membership cannot be memorised across splits.
* ``persistent``: the same rings are active in every period, as in the legacy
  dataset; used only to quantify the effect of entity memorisation.

Ground truth (fraud type, ring id) is returned in a separate ``truth`` frame
and is never part of the claims frame that feature code reads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 30


@dataclass(frozen=True)
class GeneratorConfig:
    n_patients: int = 6000
    n_doctors: int = 300
    n_hospitals: int = 40
    n_regions: int = 5
    n_specialties: int = 8
    n_periods: int = 4                 # periods 1..n-2 train, n-1 validation, n test
    period_months: int = 6
    legit_rate_mean: float = 3.0       # mean legitimate claims per patient over the horizon
    legit_rate_shape: float = 1.5      # gamma shape: patient utilisation heterogeneity
    p_regular: float = 0.85            # share of legit claims filed with a regular doctor
    isolated_rate: float = 0.03        # isolated fraud claims as a share of legit claims
    isolated_shift: float = 2.0
    ring_shift: float = 0.0            # delta
    ring_density: float = 0.5          # s
    ring_doctors: int = 4
    ring_patients: int = 12
    ring_claims_per_month: float = 16.0
    ring_shape: str = "block"          # "block", "hub" or "pairs" (see module docstring)
    rings_per_period: int = 4
    persistent_rings: int = 0          # >0 switches to the persistent protocol with this many rings
    seed: int = 0

    def with_(self, **kwargs) -> "GeneratorConfig":
        return replace(self, **kwargs)

    def as_dict(self) -> dict:
        d = asdict(self)
        if d["ring_shape"] == "block":         # the default is omitted, so earlier configurations keep their hashes
            d.pop("ring_shape")
        return d

    @property
    def protocol(self) -> str:
        return "persistent" if self.persistent_rings > 0 else "emergent"


@dataclass
class SyntheticWorld:
    config: GeneratorConfig
    patients: pd.DataFrame
    doctors: pd.DataFrame
    hospitals: pd.DataFrame
    claims: pd.DataFrame      # observable claim records, sorted by (day, claim_id)
    truth: pd.DataFrame       # claim_id, fraud_label, fraud_type, ring_id (same row order as claims)
    split: np.ndarray         # "train" / "val" / "test" per claim row
    period: np.ndarray        # audit period index per claim row

    @property
    def y(self) -> np.ndarray:
        return self.truth["fraud_label"].to_numpy()


def _specialty_params(rng: np.random.Generator, n_specialties: int) -> pd.DataFrame:
    return pd.DataFrame({
        "specialty": np.arange(n_specialties),
        "mu_log_amount": rng.uniform(6.0, 8.0, n_specialties),
        "p_inpatient": rng.uniform(0.05, 0.30, n_specialties),
        "proc_rate": rng.uniform(0.6, 1.6, n_specialties),
    })


def _claim_features(rng, spec_rows: pd.DataFrame, shift: float) -> dict:
    """Draw claim features for specialties ``spec_rows``; ``shift`` in SD units."""
    n = len(spec_rows)
    sigma = 0.6
    p_in = np.clip(spec_rows["p_inpatient"].to_numpy() * (1 + 0.5 * shift), 0, 0.95)
    inpatient = rng.random(n) < p_in
    procs = 1 + rng.poisson(spec_rows["proc_rate"].to_numpy() + 0.8 * shift)
    days = np.where(inpatient, 1 + rng.poisson(3 + shift, n), 0)
    log_amount = (spec_rows["mu_log_amount"].to_numpy() + 0.15 * (procs - 1) + 0.8 * inpatient
                  + rng.normal(0, sigma, n) + shift * sigma)
    code_offset = np.clip(rng.integers(0, 10, n) + np.round(shift * 2).astype(int), 0, 9)
    return {
        "claim_amount": np.round(np.exp(log_amount), 2),
        "num_procedures": procs,
        "days_in_hospital": days,
        "inpatient": inpatient.astype(int),
        "procedure_code": spec_rows["specialty"].to_numpy() * 10 + code_offset,
        "diagnosis_code": rng.integers(0, 40, n),
    }


def _seasonal_days(rng, n: int, lo: float, hi: float) -> np.ndarray:
    """Uniform days in [lo, hi) with a mild winter bump, via rejection sampling."""
    out = np.empty(0)
    while len(out) < n:
        cand = rng.uniform(lo, hi, 2 * n + 10)
        month = (cand // DAYS_PER_MONTH) % 12
        keep = rng.random(len(cand)) < (1 + 0.15 * np.cos(2 * np.pi * month / 12)) / 1.15
        out = np.concatenate([out, cand[keep]])
    return np.floor(out[:n]).astype(int)


def generate(config: GeneratorConfig) -> SyntheticWorld:
    rng = np.random.default_rng(config.seed)
    period_days = config.period_months * DAYS_PER_MONTH
    horizon = config.n_periods * period_days

    spec = _specialty_params(rng, config.n_specialties)
    hospitals = pd.DataFrame({
        "hospital_id": np.arange(config.n_hospitals),
        "region": rng.integers(0, config.n_regions, config.n_hospitals),
        "num_beds": np.round(np.exp(rng.normal(5.3, 0.6, config.n_hospitals))).astype(int),
    })
    doc_hosp = rng.integers(0, config.n_hospitals, config.n_doctors)
    doctors = pd.DataFrame({
        "doctor_id": np.arange(config.n_doctors),
        "hospital_id": doc_hosp,
        "region": hospitals["region"].to_numpy()[doc_hosp],
        "specialty": rng.integers(0, config.n_specialties, config.n_doctors),
        "years_experience": rng.integers(1, 41, config.n_doctors),
        "popularity": np.exp(rng.normal(0, 0.8, config.n_doctors)),
    })
    utilisation = rng.gamma(config.legit_rate_shape, config.legit_rate_mean / config.legit_rate_shape, config.n_patients)
    patients = pd.DataFrame({
        "patient_id": np.arange(config.n_patients),
        "region": rng.integers(0, config.n_regions, config.n_patients),
        "age": rng.integers(18, 91, config.n_patients),
        "gender": rng.integers(0, 2, config.n_patients),
        "chronic_count": rng.poisson(1.2, config.n_patients),
        "utilisation": utilisation,
    })

    # Ring membership is drawn first: disjoint doctors and patients per ring.
    # Recruited patients skew towards high utilisers (camouflage).
    periods = [(k * period_days, (k + 1) * period_days) for k in range(config.n_periods)]
    if config.persistent_rings > 0:
        windows = [periods] * config.persistent_rings          # same members in every period
    else:
        windows = [[per] for per in periods for _ in range(config.rings_per_period)]
    n_ring_docs, n_ring_pats = len(windows) * config.ring_doctors, len(windows) * config.ring_patients
    if n_ring_docs > config.n_doctors or n_ring_pats > config.n_patients:
        raise ValueError("Not enough doctors/patients for disjoint rings")
    doc_pool = rng.permutation(config.n_doctors)[:n_ring_docs]
    pat_pool = rng.choice(config.n_patients, size=n_ring_pats, replace=False, p=utilisation / utilisation.sum())
    ring_patient = np.zeros(config.n_patients, bool)
    ring_patient[pat_pool] = True

    # Regular doctors: 1-3 per patient (exactly 1 for ring patients, so that the
    # number of distinct doctors a ring patient sees is governed by ring_density),
    # same region, popularity weighted.
    pop = doctors["popularity"].to_numpy()
    by_region = {r: np.flatnonzero(doctors["region"].to_numpy() == r) for r in range(config.n_regions)}
    regular = []
    for r, is_ring in zip(patients["region"].to_numpy(), ring_patient):
        cand = by_region[r] if len(by_region[r]) else np.arange(config.n_doctors)
        k = 1 if is_ring else min(len(cand), 1 + rng.binomial(2, 0.5))
        regular.append(rng.choice(cand, size=k, replace=False, p=pop[cand] / pop[cand].sum()))

    records = []

    def add(pat, doc, day, shift, ftype, ring):
        rows = spec.iloc[doctors["specialty"].to_numpy()[doc]].reset_index(drop=True)
        feats = _claim_features(rng, rows, shift)
        records.append(pd.DataFrame({
            "patient_id": pat, "doctor_id": doc, "hospital_id": doctors["hospital_id"].to_numpy()[doc],
            "day": day, **feats, "fraud_type": ftype, "ring_id": ring,
        }))

    # Legitimate claims.
    n_legit_pp = rng.poisson(utilisation)
    pat = np.repeat(patients["patient_id"].to_numpy(), n_legit_pp)
    use_regular = rng.random(len(pat)) < config.p_regular
    doc = rng.choice(config.n_doctors, size=len(pat), p=pop / pop.sum())
    reg_pick = np.array([rng.choice(regular[p]) for p in pat]) if len(pat) else np.empty(0, int)
    doc = np.where(use_regular, reg_pick, doc)
    add(pat, doc, _seasonal_days(rng, len(pat), 0, horizon), 0.0, "legit", 0)

    # Isolated fraud: upcoded claims through a patient's regular doctor.
    n_iso = int(round(config.isolated_rate * len(pat)))
    iso_pat = rng.choice(config.n_patients, size=n_iso)
    iso_doc = np.array([rng.choice(regular[p]) for p in iso_pat])
    add(iso_pat, iso_doc, _seasonal_days(rng, n_iso, 0, horizon), config.isolated_shift, "isolated", 0)

    # Provider rings.
    for ring_idx, active in enumerate(windows):
        r_docs = doc_pool[ring_idx * config.ring_doctors:(ring_idx + 1) * config.ring_doctors]
        r_pats = pat_pool[ring_idx * config.ring_patients:(ring_idx + 1) * config.ring_patients]
        if config.ring_shape == "block":
            links = rng.random((len(r_pats), len(r_docs))) < config.ring_density
            links[np.arange(len(r_pats)), rng.integers(0, len(r_docs), len(r_pats))] = True   # every patient linked
            links[rng.integers(0, len(r_pats), len(r_docs)), np.arange(len(r_docs))] = True   # every doctor linked
        elif config.ring_shape == "hub":
            # Doctor 0 is the hub; spokes receive the patients in (near) equal shares.
            links = np.zeros((len(r_pats), len(r_docs)), bool)
            links[:, 0] = True
            links[np.arange(len(r_pats)), 1 + rng.permutation(len(r_pats)) % (len(r_docs) - 1)] = True
        elif config.ring_shape == "pairs":
            links = np.zeros((len(r_pats), len(r_docs)), bool)
            for i in range(len(r_pats)):
                links[i, rng.choice(len(r_docs), 2, replace=False)] = True
            for j in np.flatnonzero(~links.any(0)):                                          # every doctor linked
                links[rng.integers(0, len(r_pats)), j] = True
        else:
            raise ValueError(f"unknown ring_shape {config.ring_shape!r}")
        li, lj = np.nonzero(links)
        for start, end in active:
            n_claims = int(round(config.ring_claims_per_month * (end - start) / DAYS_PER_MONTH))
            pick = rng.integers(0, len(li), n_claims)
            # Lockstep timing: claims cluster around burst days (about one per month).
            bursts = rng.uniform(start, end, max(2, int(round((end - start) / DAYS_PER_MONTH))))
            day = np.clip(np.floor(rng.choice(bursts, n_claims) + rng.poisson(2, n_claims)), start, end - 1).astype(int)
            add(r_pats[li[pick]], r_docs[lj[pick]], day, config.ring_shift, "ring", ring_idx + 1)

    claims = pd.concat(records, ignore_index=True)
    claims = claims.sort_values(["day"], kind="stable").reset_index(drop=True)
    claims.insert(0, "claim_id", np.arange(len(claims)))
    truth = pd.DataFrame({
        "claim_id": claims["claim_id"],
        "fraud_type": claims.pop("fraud_type"),
        "ring_id": claims.pop("ring_id").astype(int),
    })
    truth["fraud_label"] = (truth["fraud_type"] != "legit").astype(int)
    period = (claims["day"].to_numpy() // period_days).astype(int)
    split = np.where(period < config.n_periods - 2, "train", np.where(period == config.n_periods - 2, "val", "test"))
    return SyntheticWorld(config, patients.drop(columns=["utilisation"]), doctors.drop(columns=["popularity"]),
                          hospitals, claims, truth, split, period)
