from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import torch


ROOT = Path(__file__).resolve().parent


@dataclass
class BestRun:
	approach: str
	f1: float
	timestamp: str
	results_file: Path


@dataclass
class ParamReport:
	approach: str
	f1: float
	timestamp: str
	results_file: Path
	artifact: str
	total_params: int | None
	learnable_params: int | None
	notes: str


def parse_timestamp(text: str) -> str | None:
	m = re.search(r"(?im)^\s*Timestamp\s*:?\s*([0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2})", text)
	return m.group(1) if m else None


def parse_best_runs(root: Path) -> dict[str, BestRun]:
	candidates: list[BestRun] = []
	for p in root.rglob("results*.txt"):
		if "old" in p.parts:
			continue

		text = p.read_text(encoding="utf-8", errors="ignore")
		timestamp = parse_timestamp(text)
		if not timestamp:
			continue

		is_hybrid_file = (
			re.search(r"(?im)^\s*Rule-Based Chunker:", text)
			and re.search(r"(?im)^\s*Hybrid \(ML\) Chunker:", text)
		)

		if is_hybrid_file:
			rule_match = re.search(
				r"(?ims)Rule-Based Chunker:.*?^\s*F1\s*:\s*([0-9]+(?:\.[0-9]+)?)",
				text,
			)
			if rule_match:
				candidates.append(
					BestRun(
						approach="Rule-Based",
						f1=float(rule_match.group(1)),
						timestamp=timestamp,
						results_file=p,
					)
				)

			hy_match = re.search(
				r"(?ims)Hybrid \(ML\) Chunker:.*?^\s*F1\s*:\s*([0-9]+(?:\.[0-9]+)?)",
				text,
			)
			if hy_match:
				is_crf = bool(
					re.search(r"results_CRF_", p.name)
					or re.search(r"(?im)^\s*Hybrid backend\s*:\s*CRF\b", text)
				)
				approach = (
					"Hybrid (Rule + CRF)"
					if is_crf
					else "Hybrid (Rule + Logistic Regression)"
				)
				candidates.append(
					BestRun(
						approach=approach,
						f1=float(hy_match.group(1)),
						timestamp=timestamp,
						results_file=p,
					)
				)
			continue

		f1_match = re.search(r"(?im)^\s*F1\s*:\s*([0-9]+(?:\.[0-9]+)?)", text)
		if not f1_match:
			continue

		name = p.name
		if re.search(r"results_Mamba_", name):
			approach = "Mamba"
		elif re.search(r"results_UT-CRF_|results_UT_", name):
			approach = "UT-CRF"
		elif re.search(r"results_BiLSTM-CRF_paper_style_", name):
			approach = "BiLSTM-CRF (paper-style)"
		elif re.search(r"results_BiLSTM-CRF_", name):
			approach = "BiLSTM-CRF (baseline)"
		elif re.search(r"results_BiLSTM_", name):
			approach = "BiLSTM"
		elif re.search(r"results_Transformer_", name):
			# Pretrained Transformer encoder + (optional) CRF head
			approach = "Transformer-CRF"
		else:
			continue

		candidates.append(
			BestRun(
				approach=approach,
				f1=float(f1_match.group(1)),
				timestamp=timestamp,
				results_file=p,
			)
		)

	best: dict[str, BestRun] = {}
	for c in candidates:
		cur = best.get(c.approach)
		if cur is None or c.f1 > cur.f1:
			best[c.approach] = c
	return best


def first_match(root: Path, pattern: str) -> Path | None:
	matches = sorted(root.rglob(pattern))
	return matches[0] if matches else None


def find_artifact(root: Path, run: BestRun) -> Path | None:
	ts = run.timestamp
	ap = run.approach

	if ap == "BiLSTM-CRF (paper-style)":
		return first_match(root, f"**/*paper_style*{ts}*.pt")
	if ap == "BiLSTM-CRF (baseline)":
		matches = [
			p
			for p in root.rglob(f"**/*BiLSTM-CRF*{ts}*.pt")
			if "paper_style" not in p.name
		]
		return sorted(matches)[0] if matches else None
	if ap == "BiLSTM":
		return first_match(root, f"**/*BiLSTM_{ts}*.pt")
	if ap == "Mamba":
		return first_match(root, f"**/*Mamba_{ts}*.pt")
	if ap == "UT-CRF":
		return first_match(root, f"**/*UT-CRF_{ts}*.pt")
	if ap == "Transformer-CRF":
		# e.g. transformer_crf_chunker_Transformer_2026-04-08_14-06-50.pt
		return first_match(root, f"**/*transformer_crf_chunker_*{ts}*.pt")
	if ap == "Hybrid (Rule + CRF)":
		return first_match(root, f"**/*hybrid_chunker_CRF_{ts}.joblib")
	if ap == "Hybrid (Rule + Logistic Regression)":
		return first_match(root, f"**/*hybrid_chunker_{ts}.joblib")
	if ap == "Rule-Based":
		return None
	return None


def count_pt_params(path: Path) -> tuple[int, int, str]:
	data = torch.load(path, map_location="cpu")

	if isinstance(data, dict) and "model_state_dict" in data and isinstance(data["model_state_dict"], dict):
		state_dict = data["model_state_dict"]
	elif isinstance(data, dict) and all(torch.is_tensor(v) for v in data.values()):
		state_dict = data
	else:
		raise ValueError("Unsupported checkpoint format; expected state_dict-like content.")

	tensors = [v for v in state_dict.values() if torch.is_tensor(v)]
	total = sum(t.numel() for t in tensors)

	# Loaded state_dict tensors typically have requires_grad=False. For model checkpoints,
	# parameter tensors in state_dict represent trainable model weights.
	learnable = sum(t.numel() for t in tensors if bool(getattr(t, "requires_grad", False)))
	note = "learnable inferred from state_dict"
	if learnable == 0 and total > 0:
		learnable = total
		note = "state_dict tensors default to requires_grad=False; learnable approximated as total"

	return total, learnable, note


def count_joblib_params(path: Path) -> tuple[int | None, int | None, str]:
	model = joblib.load(path)

	# Logistic Regression backend wrapped by HybridChunkCorrector
	clf = getattr(model, "classifier", None)
	if clf is not None and hasattr(clf, "coef_"):
		total = int(clf.coef_.size)
		if hasattr(clf, "intercept_"):
			total += int(clf.intercept_.size)
		return total, total, "counted from LogisticRegression coef_ + intercept_"

	# CRF backend wrapped by HybridChunkCorrector
	crf = getattr(model, "crf", None)
	if crf is not None:
		state_features = getattr(crf, "state_features_", None)
		transition_features = getattr(crf, "transition_features_", None)
		sf = len(state_features) if state_features is not None else 0
		tf = len(transition_features) if transition_features is not None else 0
		total = sf + tf
		return total, total, "counted from CRF state_features_ + transition_features_"

	return None, None, "unknown joblib model structure"


def build_reports(root: Path) -> list[ParamReport]:
	best = parse_best_runs(root)
	ordered_approaches = [
		"Rule-Based",
		"Hybrid (Rule + Logistic Regression)",
		"Hybrid (Rule + CRF)",
		"BiLSTM",
		"BiLSTM-CRF (baseline)",
		"BiLSTM-CRF (paper-style)",
		"Mamba",
		"UT-CRF",
		"Transformer-CRF",
	]

	reports: list[ParamReport] = []
	for approach in ordered_approaches:
		run = best.get(approach)
		if run is None:
			continue

		artifact = find_artifact(root, run)
		if approach == "Rule-Based":
			reports.append(
				ParamReport(
					approach=approach,
					f1=run.f1,
					timestamp=run.timestamp,
					results_file=run.results_file,
					artifact="N/A",
					total_params=0,
					learnable_params=0,
					notes="rule-based system has no trainable parameters",
				)
			)
			continue

		if artifact is None:
			reports.append(
				ParamReport(
					approach=approach,
					f1=run.f1,
					timestamp=run.timestamp,
					results_file=run.results_file,
					artifact="NOT FOUND",
					total_params=None,
					learnable_params=None,
					notes="best-run artifact not found by timestamp",
				)
			)
			continue

		try:
			if artifact.suffix.lower() == ".pt":
				total, learnable, notes = count_pt_params(artifact)
			elif artifact.suffix.lower() == ".joblib":
				total, learnable, notes = count_joblib_params(artifact)
			else:
				total, learnable, notes = None, None, "unsupported artifact type"
		except Exception as exc:  # pragma: no cover - defensive reporting
			total, learnable, notes = None, None, f"error while reading artifact: {exc}"

		reports.append(
			ParamReport(
				approach=approach,
				f1=run.f1,
				timestamp=run.timestamp,
				results_file=run.results_file,
				artifact=str(artifact),
				total_params=total,
				learnable_params=learnable,
				notes=notes,
			)
		)

	return reports


def fmt_int(value: int | None) -> str:
	return "N/A" if value is None else f"{value:,}"


def main() -> None:
	reports = build_reports(ROOT)
	if not reports:
		print("No best-model reports found.")
		return

	print("BEST MODELS: TOTAL + LEARNABLE PARAMETERS\n")
	header = (
		f"{'Approach':36} {'Best F1':>8} {'Total Params':>14} "
		f"{'Learnable Params':>18}"
	)
	print(header)
	print("-" * len(header))
	for r in reports:
		print(
			f"{r.approach:36} {r.f1:8.4f} {fmt_int(r.total_params):>14} "
			f"{fmt_int(r.learnable_params):>18}"
		)

	print("\nDETAILS")
	print("-" * 80)
	for r in reports:
		print(f"Approach     : {r.approach}")
		print(f"Best F1      : {r.f1:.4f}")
		print(f"Timestamp    : {r.timestamp}")
		print(f"Results file : {r.results_file}")
		print(f"Artifact     : {r.artifact}")
		print(f"Total params : {fmt_int(r.total_params)}")
		print(f"Learnable    : {fmt_int(r.learnable_params)}")
		print(f"Notes        : {r.notes}")
		print("-" * 80)


if __name__ == "__main__":
	main()