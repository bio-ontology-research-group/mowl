"""Published MRR (and Hits@3) of Query2Box, BetaE, ConE and AConE on the released query
datasets of Query2Box / BetaE.

Transcribed from the LaTeX sources of the arXiv versions (values in the papers are given in
percent, except Query2Box's, which are fractions; all are stored as fractions). Only the
rows of methods mOWL implements are recorded; rows of GQE, LinE and the variants of a paper
are not. Subsets are the structure names, ``average_epfo`` (the "Avg"/"AVG" column of the
nine positive structures) and ``average_negation`` (the "AVG" of the five structures with
negation). 2u and up are the DNF versions (BetaE's Table 1 also gives a De Morgan variant,
not recorded).

Data sets and tables:

* Query2Box queries (``query-*-q2b`` benchmarks):
  Ren et al. 2020 (arXiv:2002.05969), H@3 in the main-text table "H@3 results of Query2box
  vs. GQE" (LaTeX label ``tab:hits-baseline``) and MRR in the appendix table "MRR results of
  Query2box vs. GQE" (``tab:mrr-baseline``), both for Query2Box at d = 400; BetaE's appendix
  table "MRR results (%) on queries from Ren et al. 2020" (``tab:conjold``) gives Query2Box
  and BetaE re-run by the BetaE authors on these queries.
* BetaE queries (``query-*-betae`` benchmarks):
  BetaE (arXiv:2010.11465), Table 1 (MRR on EPFO queries; BetaE and Query2Box, 400
  dimensions) and Table 2 (MRR and H@10 on queries with negation; BetaE);
  ConE (arXiv:2110.13715), Table 1 (MRR on EPFO queries; ConE d = 800, Query2Box retrained
  with d = 1600; the BetaE rows are copied from BetaE) and Table 2 (MRR with negation;
  ConE); AConE (arXiv:2407.09212), Table 1 for NELL-QA, which is the BetaE NELL995 data.
* WN18RR-QA (``query-wn18rr-qa``; data of Huang et al., 2022, not in the BetaE archive):
  AConE Table 1; its BetaE, Query2Box and LinE rows are quoted from the original papers and
  ConE on WN18RR is the authors' own re-run (Section 6.1 of the paper).

The averages are those printed in the papers (not recomputed). A published average of the
nine positive structures is the mean of the per-structure MRRs, as in mOWL's
``average_epfo``.
"""

from mowl.benchmarks.spec import PublishedResult
from mowl.benchmarks.entries._refs import (HE2024ACONE, REN2020BETAE, REN2020Q2B,
                                           ZHANG2021CONE)

_EPFO = ("1p", "2p", "3p", "2i", "3i", "pi", "ip", "2u", "up")     # column order, BetaE/ConE
_EPFO_Q2B = ("1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up")  # column order, Query2Box
_NEG = ("2in", "3in", "inp", "pin", "pni")


def _numbers(order, text, average=None, scale=0.01):
    values = {n: round(float(v) * scale, 6) for n, v in zip(order, text.split()) if v != "-"}
    out = dict(values)
    if average is not None:
        group = "average_epfo" if order[0] == "1p" else "average_negation"
        out[group] = round(float(average) * scale, 6)
    return out


def _add(rows, benchmark, method, source, locator, metric, order, text, average=None,
         scale=0.01, settings=None, note=""):
    rows.append((benchmark, method, source, locator, metric,
                 _numbers(order, text, average, scale), settings or {}, note))


RESULTS = []
_ROWS = []
_BETAE_NOTE = ("Trained by the BetaE authors on the released training queries (the five "
               "positive structures for Query2Box; those and the five negation structures "
               "for BetaE).")

# -- BetaE paper: Table 1 (EPFO MRR) -------------------------------------------------------
for _kg, _betae, _q2b in (
        ("fb15k", ("65.1 25.7 24.7 55.8 66.5 43.9 28.1 40.1 25.2", 41.6),
         ("68.0 21.0 14.2 55.1 66.5 39.4 26.1 35.1 16.7", 38.0)),
        ("fb15k237", ("39.0 10.9 10.0 28.8 42.5 22.4 12.6 12.4 9.7", 20.9),
         ("40.6 9.4 6.8 29.5 42.3 21.2 12.6 11.3 7.6", 20.1)),
        ("nell995", ("53.0 13.0 11.4 37.6 47.5 24.1 14.3 12.2 8.5", 24.6),
         ("42.2 14.0 11.2 33.3 44.5 22.4 16.8 11.3 10.3", 22.9))):
    _add(_ROWS, f"query-{_kg}-betae", "betae", REN2020BETAE, "Table 1 (MRR, EPFO queries)",
         "mrr", _EPFO, _betae[0], _betae[1], settings={"embed_dim": 400}, note=_BETAE_NOTE)
    _add(_ROWS, f"query-{_kg}-betae", "query2box", REN2020BETAE,
         "Table 1 (MRR, EPFO queries)", "mrr", _EPFO, _q2b[0], _q2b[1],
         settings={"embed_dim": 400}, note="Query2Box re-run by the BetaE authors. " + _BETAE_NOTE)

# -- BetaE paper: Table 2 (negation) --------------------------------------------------------
for _kg, _mrr, _avg, _h10, _havg in (
        ("fb15k", "14.3 14.7 11.5 6.5 12.4", 11.8, "30.8 31.9 23.4 14.3 26.3", 25.3),
        ("fb15k237", "5.1 7.9 7.4 3.6 3.4", 5.4, "11.3 17.3 16.0 8.1 7.0", 11.9),
        ("nell995", "5.1 7.8 10.0 3.1 3.5", 5.9, "11.6 18.2 20.8 6.9 7.2", 12.9)):
    _add(_ROWS, f"query-{_kg}-betae", "betae", REN2020BETAE, "Table 2 (MRR, negation)", "mrr",
         _NEG, _mrr, _avg, settings={"embed_dim": 400}, note=_BETAE_NOTE)
    _add(_ROWS, f"query-{_kg}-betae", "betae", REN2020BETAE, "Table 2 (H@10, negation)",
         "hits@10", _NEG, _h10, _havg, settings={"embed_dim": 400}, note=_BETAE_NOTE)

# -- BetaE paper: appendix, MRR on the Query2Box queries ------------------------------------
for _kg, _betae, _q2b in (
        ("fb15k", ("65.0 42.1 37.8 52.9 64.0 41.5 22.9 48.8 26.9", 44.6),
         ("67.1 38.0 27.5 49.2 62.8 36.2 19.2 49.0 28.9", 42.0)),
        ("fb15k237", ("39.1 24.2 20.4 28.1 39.2 19.4 10.6 22.0 17.0", 24.4),
         ("40.3 22.8 17.5 27.5 37.9 18.5 10.5 20.5 17.4", 23.6)),
        ("nell995", ("53.0 27.5 28.1 32.9 45.1 21.8 10.4 38.6 19.6", 30.7),
         ("41.8 22.9 20.8 28.6 41.2 19.9 12.3 26.9 15.5", 25.5))):
    _loc = "Appendix, MRR on queries from Ren et al. 2020 (tab:conjold)"
    _add(_ROWS, f"query-{_kg}-q2b", "betae", REN2020BETAE, _loc, "mrr", _EPFO, _betae[0],
         _betae[1], settings={"embed_dim": 400},
         note="BetaE trained on the training queries (shared by the two releases) and tested "
              "on the Query2Box validation and test queries.")
    _add(_ROWS, f"query-{_kg}-q2b", "query2box", REN2020BETAE, _loc, "mrr", _EPFO, _q2b[0],
         _q2b[1], settings={"embed_dim": 400}, note="Query2Box re-run by the BetaE authors.")

# -- Query2Box paper: MRR and H@3 on its own queries ---------------------------------------
for _kg, _mrr, _h3 in (
        ("fb15k", ("0.654 0.373 0.274 0.488 0.602 0.194 0.339 0.468 0.301", 0.41),
         ("0.786 0.413 0.303 0.593 0.712 0.211 0.397 0.608 0.33", 0.484)),
        ("fb15k237", ("0.4 0.225 0.173 0.275 0.378 0.105 0.18 0.198 0.178", 0.235),
         ("0.467 0.24 0.186 0.324 0.453 0.108 0.205 0.239 0.193", 0.268)),
        ("nell995", ("0.413 0.227 0.208 0.288 0.414 0.125 0.193 0.266 0.155", 0.254),
         ("0.555 0.266 0.233 0.343 0.48 0.132 0.212 0.369 0.163", 0.306))):
    _s = {"embed_dim": 400, "gamma": 24.0}
    _add(_ROWS, f"query-{_kg}-q2b", "query2box", REN2020Q2B,
         "Appendix, MRR results (tab:mrr-baseline)", "mrr", _EPFO_Q2B, _mrr[0], _mrr[1],
         scale=1.0, settings=_s)
    _add(_ROWS, f"query-{_kg}-q2b", "query2box", REN2020Q2B,
         "Main text, H@3 results (tab:hits-baseline)", "hits@3", _EPFO_Q2B, _h3[0], _h3[1],
         scale=1.0, settings=_s)

# -- ConE paper: Tables 1 and 2 -------------------------------------------------------------
for _kg, _cone, _q2b, _neg in (
        ("fb15k", ("73.3 33.8 29.2 64.4 73.7 50.9 35.7 55.7 31.4", 49.8),
         ("70.5 23.0 15.1 61.2 71.8 41.8 28.7 37.7 19.0", 40.1),
         ("17.9 18.7 12.5 9.8 15.1", 14.8)),
        ("fb15k237", ("41.8 12.8 11.0 32.6 47.3 25.5 14.0 14.5 10.8", 23.4),
         ("41.3 9.9 7.2 31.1 45.4 21.9 13.3 11.9 8.1", 21.1), ("5.4 8.6 7.8 4.0 3.6", 5.9)),
        ("nell995", ("53.1 16.1 13.9 40.0 50.8 26.3 17.5 15.3 11.3", 27.2),
         ("42.7 14.5 11.7 34.7 45.8 23.2 17.4 12.0 10.7", 23.6), ("5.7 8.1 10.8 3.5 3.9", 6.4))):
    _add(_ROWS, f"query-{_kg}-betae", "cone_qe", ZHANG2021CONE, "Table 1 (MRR, EPFO queries)",
         "mrr", _EPFO, _cone[0], _cone[1], settings={"embed_dim": 800},
         note="Mean of five seeds.")
    _add(_ROWS, f"query-{_kg}-betae", "cone_qe", ZHANG2021CONE, "Table 2 (MRR, negation)",
         "mrr", _NEG, _neg[0], _neg[1], settings={"embed_dim": 800}, note="Mean of five seeds.")
    _add(_ROWS, f"query-{_kg}-betae", "query2box", ZHANG2021CONE,
         "Table 1 (MRR, EPFO queries)", "mrr", _EPFO, _q2b[0], _q2b[1],
         settings={"embed_dim": 1600},
         note="The ConE authors retrained the baselines with d = 800 and report Query2Box "
              "(and GQE) with d = 1600 and BetaE with d = 400 (Section 5.1).")

# -- AConE paper: Table 1 --------------------------------------------------------------------
_ACONE_ORDER = _EPFO + _NEG
for _bench, _rows in (
        ("query-nell995-betae",
         (("acone", "54.5 17.7 14.4 41.9 53.0 26.1 20.7 16.5 12.8 5.2 7.7 9.4 3.2 3.7", ""),)),
        ("query-wn18rr-qa",
         (("acone", "50.9 17.6 9.9 70.5 89.0 38.9 29.6 18.4 14.0 18.3 70.1 13.4 7.6 10.4", ""),
          ("cone_qe", "46.8 14.5 9.3 59.0 83.9 33.6 18.7 10.0 9.8 13.9 61.8 10.6 7.3 7.6",
           "ConE re-run by the AConE authors with a hyperparameter search (not published "
           "before)."),
          ("betae", "44.1 9.8 3.8 57.2 76.2 32.6 17.9 7.5 5.3 12.7 59.9 5.1 4.0 7.4",
           "Quoted by the AConE paper from the original paper."),
          ("query2box", "22.4 4.6 2.3 25.6 41.2 13.2 11.0 2.9 3.4 - - - - -",
           "Quoted by the AConE paper from the original paper.")))):
    for _method, _text, _note in _rows:
        _add(_ROWS, _bench, _method, HE2024ACONE, "Table 1 (MRR)", "mrr", _ACONE_ORDER, _text,
             settings={"embed_dim": 800} if _method == "acone" else {}, note=_note)
        # the table prints no average

for _b, _m, _src, _loc, _metric, _vals, _settings, _note in _ROWS:
    for _subset, _value in _vals.items():
        RESULTS.append(PublishedResult(
            benchmark=_b, method=_m, metrics={_metric: _value}, source=_src, locator=_loc,
            subset=_subset, settings=_settings, note=_note))
