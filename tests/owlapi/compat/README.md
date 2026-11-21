# Compatibility tests for the Java libraries of mOWL

mOWL calls the OWL API and libraries built on it (ELK, HermiT, the KM reasoner, jcel,
mOWL's own `gateway.jar`) through JPype. An upgrade of any of them can change what mOWL
computes without any Python error. These tests pin the behaviour.

## What is here

- `fixtures/el.ofn`: an EL++ ontology (existentials, conjunctions, bottom, disjointness,
  role inclusions, a role chain, a transitive role, a domain, ABox, annotations).
  `fixtures/rich.ofn`: constructs beyond EL (negation, union, universal and cardinality
  restrictions, nominals, self, inverse and other role characteristics, data properties,
  negative assertions, annotations with language tags and datatypes). Two fixtures of
  the geometric tests (`t3`, `roles`) are probed as well.
- `probes.py`: the behaviour probes. Each returns a JSON value built from IRIs and
  counts, never from `toString()` of an OWL API object. They cover: signature and axiom
  counts after loading; annotations; detected expressivity and the per-method
  expressivity reports; EL normalisation; role axiom extraction; what an EL dataset
  contains; all graph projections; the categorical projection; the text corpora; the
  four reasoner backends (classification and realisation); automatic reasoner
  selection; the subsumption oracle on class expressions built at query time;
  `MOWLReasoner`; and saving and reloading in five syntaxes.
- `golden/<ontology>.<probe>.json`: the values recorded with the OWL API version named
  inside each file (4.5.29, the last OWL API 4 that mOWL shipped).
- `generate_golden.py`: records golden files. It does not overwrite existing ones.

`tests/owlapi/test_compat_golden.py` compares the current behaviour with the golden
files. `tests/owlapi/test_java_stack.py` checks the library set itself: the OWL API
generation, one version of every artifact, one copy of the core classes on the class
path, every Java import in mOWL's source resolves, all reasoner backends load, and a
fresh process runs them without Java linkage errors.

## Rules for an upgrade

1. Record the golden files **before** touching `mowl/lib`, and commit them.
2. Upgrade until `tests/owlapi`, `tests/reasoning`, `tests/ontology`,
   `tests/projection`, `tests/corpus`, `tests/datasets` and then the whole suite and
   `make doctest` pass.
3. A golden file may change only when the new behaviour is right and the old one was
   wrong or arbitrary (for example the order of operands in a rendered sentence). Prefer
   making mOWL independent of the difference (sort, canonicalise). Every changed golden
   file is listed below with the reason. Never regenerate to make a test pass.
4. A probe may not be weakened or a case added to `SKIP` without the same kind of entry.

## Golden files changed since they were recorded

None.

## Known limits of the reference behaviour

- `rich.ofn` has no `HasKey` axiom: the KM reasoner refuses ontologies with keys
  ("out of fragment: DL-safe rules"), and the automatic selection still picks it.
- ELK is not probed on `rich.ofn` (outside its fragment).
- The KM reasoner (1.6.5) does not reliably answer on `rich.ofn` with OWL API 4.5.29. It
  classified the ontology once (`golden/rich.reasoner_km.json`, kept as a record) and
  then never returned in every later attempt, at start-up or after the first change of
  the ontology: its helper process sits idle while the Java side waits in
  `Classifier$Session.exchange`, which has no time limit. The case is therefore in
  `probes.UNSTABLE` and not compared, and the query-time entailment probe uses HermiT on
  this ontology. On `el.ofn`, `t3.owl` and `roles.owl` KM answers and is compared; these
  cases run under a watchdog that aborts the test run after 10 minutes without an
  answer. An upgrade must not make KM worse on these three; fixing the behaviour on
  `rich.ofn` is a separate task.
