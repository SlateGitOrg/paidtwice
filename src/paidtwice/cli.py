"""Command line interface: generate, vendors, candidates, fit, score, evaluate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from paidtwice import data as files
from paidtwice.candidates import build_candidates, load_candidates, write_candidates
from paidtwice.compare import FIELDS, LEVELS
from paidtwice.em import FSModel
from paidtwice.evaluate import Truth, evaluate, render_markdown
from paidtwice.fit import fit_model
from paidtwice.generate import GeneratorConfig, generate_dataset, write_dataset
from paidtwice.score import rank_pairs
from paidtwice.vendors import cluster_vendors

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Duplicate payment detection by record linkage.",
)
DataDir = Annotated[Path, typer.Option("--data", "-d", help="Working directory for all files.")]


@app.command()
def generate(
    data: DataDir = Path("data"),
    seed: Annotated[int, typer.Option(help="Seed of the synthetic data.")] = 42,
) -> None:
    """Write vendors.csv, payments.csv and truth.csv (truth is read only by evaluate)."""
    dataset = generate_dataset(GeneratorConfig(seed=seed))
    write_dataset(dataset, data)
    typer.echo(
        f"wrote {len(dataset.vendors)} vendor records and {len(dataset.payments)} payments "
        f"to {data}/ (seed {seed})"
    )


@app.command()
def vendors(data: DataDir = Path("data")) -> None:
    """Cluster vendor records by tax ID, bank account, or normalised name and ZIP."""
    frame = files.read_csv(data / files.VENDORS)
    clusters = cluster_vendors(frame)
    clusters.to_csv(data / files.CLUSTERS, index=False)
    sizes = clusters.drop_duplicates("cluster_id")["cluster_size"]
    typer.echo(
        f"{len(clusters)} vendor records -> {len(sizes)} clusters; "
        f"{int((sizes > 1).sum())} clusters hold more than one record "
        f"({int(sizes[sizes > 1].sum())} records)"
    )


@app.command()
def candidates(data: DataDir = Path("data")) -> None:
    """Block the payments into candidate pairs and report the reduction."""
    p = files.load_payments(data)
    cand = build_candidates(p)
    write_candidates(cand, p, data)
    (data / files.BLOCKING_REPORT).write_text(json.dumps(cand.report, indent=2) + "\n")
    r = cand.report
    typer.echo(f"payments: {r['payments']:,}   all pairs: {r['all_pairs']:,}")
    for name, count in r["pairs_per_block"].items():
        only = r["pairs_only_in_block"][name]
        typer.echo(f"  block {name:<15} {count:>9,} pairs  ({only:,} found only here)")
    typer.echo(
        f"candidate pairs: {r['candidate_pairs']:,} "
        f"({100 * r['share_of_all_pairs']:.4f}% of all pairs, "
        f"reduction ratio {r['reduction_ratio']:.6f})"
    )


@app.command()
def fit(data: DataDir = Path("data")) -> None:
    """Learn Fellegi-Sunter m and u probabilities by EM and write model.json."""
    p = files.load_payments(data)
    cand = load_candidates(p, data)
    model = fit_model(p, cand)
    model.save(data / files.MODEL)
    weights = model.log2_weights()
    typer.echo(
        f"EM: {model.iterations} iterations, converged={model.converged}, "
        f"pi={model.pi:.3e}, training pairs={model.info['training_pairs']:,} "
        f"of {model.info['candidate_pairs']:,} candidates"
    )
    if model.info["pooled_fields"]:
        typer.echo(f"ordinal weights pooled for: {', '.join(model.info['pooled_fields'])}")
    typer.echo(f"{'field':<8}{'level':<17}{'m':>10}{'u':>12}{'log2 weight':>13}")
    for f in FIELDS:
        for k, level in enumerate(LEVELS[f]):
            typer.echo(
                f"{f:<8}{level:<17}{model.m[f][k]:>10.4f}{model.u[f][k]:>12.6f}{weights[f][k]:>13.2f}"
            )


@app.command()
def score(
    data: DataDir = Path("data"),
    budget: Annotated[int, typer.Option(help="Pairs the analyst can review.")] = 200,
    no_suppress: Annotated[
        bool, typer.Option("--no-suppress", help="Disable the suppression rules (ablation).")
    ] = False,
) -> None:
    """Rank candidate pairs into a review queue CSV."""
    p = files.load_payments(data)
    cand = load_candidates(p, data)
    model = FSModel.load(data / files.MODEL)
    result = rank_pairs(p, cand, model, budget=budget, suppress=not no_suppress)
    name = files.QUEUE_NO_SUPPRESS if no_suppress else files.QUEUE
    result.queue.to_csv(data / name, index=False)
    if not no_suppress:
        result.suppressed.to_csv(data / files.SUPPRESSED, index=False)
    s = result.summary
    typer.echo(
        f"wrote {name}: {s['queue_size']} pairs, ${s['amount_at_risk_in_queue']:,.2f} at risk"
    )
    typer.echo(f"candidate pairs: {s['candidate_pairs']:,}   suppression: {s['suppression']}")
    for rule, count in s["suppressed_by_rule"].items():
        typer.echo(f"  {rule:<20} {count:>6,} pairs")
    if not no_suppress:
        typer.echo(
            f"  total suppressed     {s['pairs_suppressed']:>6,} pairs -> {files.SUPPRESSED}"
        )


@app.command(name="evaluate")
def evaluate_command(
    data: DataDir = Path("data"),
    budget: Annotated[int, typer.Option(help="Review budget to evaluate.")] = 200,
) -> None:
    """Compare the queues with the planted truth and write evaluation.md."""
    p = files.load_payments(data)
    cand = load_candidates(p, data)
    queue = files.read_csv(data / files.QUEUE)
    no_suppress_path = data / files.QUEUE_NO_SUPPRESS
    ablation = files.read_csv(no_suppress_path) if no_suppress_path.exists() else None
    result = evaluate(p, Truth.load(data), cand, queue, ablation, budget)
    text = render_markdown(result)
    (data / files.EVALUATION).write_text(text)
    typer.echo(text)


if __name__ == "__main__":
    app()
