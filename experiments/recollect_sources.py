import json, random, time, hashlib, sys, traceback
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from weightmf.config import TaskConfig
from weightmf.data import make_modular_addition
from weightmf.model import TaskTransformer
from weightmf.train import seed_everything, evaluate
from weightmf.transformer_weight_dataset import (
    transformer_weight_snapshot,
    TRANSFORMER_WEIGHT_NAMES,
    canonical_row_layout,
)

BASE = Path(__file__).resolve().parents[1]
OLD = BASE / "configs/transition/split_manifest.json"
SOURCE = BASE / "data/source_transition"
OUT = BASE / "data/source_stratified"


def replace_retry(src, dst):
    for i in range(40):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if i == 39:
                raise
            time.sleep(0.25)


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    replace_retry(tmp, path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads(OLD.read_text())
    seeds = split["train_seeds"] + split["validation_seeds"] + split["test_seeds"]
    assert len(seeds) == 602 and len(set(seeds)) == 602
    write(OUT / "split_manifest.json", split)
    old_manifest = json.loads((SOURCE / "manifest.json").read_text())
    shards = {x["seed"]: x for x in old_manifest["runs"]}
    existing = (
        json.loads((OUT / "manifest.json").read_text())
        if (OUT / "manifest.json").exists()
        else {"runs": []}
    )
    prior = {x["seed"]: x for x in existing["runs"]}
    cfg = TaskConfig()
    results = []
    for idx, seed in enumerate(seeds):
        path = OUT / f"transformer_pairs_seed_{seed}.pt"
        if path.exists():
            results.append(
                prior.get(
                    seed,
                    dict(
                        seed=seed,
                        file=path.name,
                        bytes=path.stat().st_size,
                        wall_seconds=None,
                    ),
                )
            )
            continue
        old = torch.load(
            SOURCE / shards[seed]["file"], map_location="cpu", weights_only=False
        )
        steps = old["steps"]
        gaps = [int(steps[j] - steps[i]) for i, j in old["pair_indices"].tolist()]
        end = min(cfg.max_steps, int(old["threshold_step"]) + 2000)
        rng = random.Random(46000000 + seed)
        pairs = []
        for k, gap in enumerate(gaps):
            # One draw per equal quantile of the feasible start interval.
            max_index = (end - gap) // 25
            lo = (k * (max_index + 1)) // 10
            hi = max(lo + 1, ((k + 1) * (max_index + 1)) // 10)
            start = 25 * rng.randrange(lo, hi)
            pairs.append((start, start + gap))
        wanted = set(z for pair in pairs for z in pair)
        seed_everything(seed)
        data = make_modular_addition(cfg, split_seed=seed)
        tx, ty = data.train_x.cuda(), data.train_y.cuda()
        vx, vy = data.val_x.cuda(), data.val_y.cuda()
        model = TaskTransformer(cfg).cuda()
        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        captures = {}
        started = time.perf_counter()

        def save(step):
            model.eval()
            tl, ta = evaluate(model, tx, ty)
            vl, va = evaluate(model, vx, vy)
            captures[step] = (transformer_weight_snapshot(model), tl, ta, vl, va)

        if 0 in wanted:
            save(0)
        for step in range(1, end + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(tx), ty)
            loss.backward()
            opt.step()
            if step in wanted:
                save(step)
            if step % 1000 == 0:
                write(
                    OUT / "status.json",
                    dict(
                        state="training",
                        completed=idx,
                        total=len(seeds),
                        seed=seed,
                        step=step,
                        end=end,
                    ),
                )
        keep = sorted(captures)
        lookup = {s: i for i, s in enumerate(keep)}
        payload = dict(
            format_version=1,
            scope="transformer_block_weights",
            seed=seed,
            threshold_step=old["threshold_step"],
            memorization_step=old["memorization_step"],
            steps=torch.tensor(keep, dtype=torch.int32),
            weights={
                n: torch.stack([captures[s][0][n] for s in keep])
                for n in TRANSFORMER_WEIGHT_NAMES
            },
            weight_names=list(TRANSFORMER_WEIGHT_NAMES),
            canonical_row_layout=canonical_row_layout(),
            pair_indices=torch.tensor(
                [[lookup[a], lookup[b]] for a, b in pairs], dtype=torch.int16
            ),
            pair_types=["start_stratified"] * 10,
            train_losses=torch.tensor([captures[s][1] for s in keep]),
            train_accuracies=torch.tensor([captures[s][2] for s in keep]),
            validation_losses=torch.tensor([captures[s][3] for s in keep]),
            validation_accuracies=torch.tensor([captures[s][4] for s in keep]),
            task_config=cfg.to_dict(),
            sampling_policy="one start per feasible-time decile, original per-run gaps",
            source_shard=shards[seed]["file"],
            wall_seconds=time.perf_counter() - started,
        )
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        replace_retry(tmp, path)
        results.append(
            dict(
                seed=seed,
                file=path.name,
                bytes=path.stat().st_size,
                wall_seconds=payload["wall_seconds"],
            )
        )
        write(
            OUT / "manifest.json",
            dict(
                policy=payload["sampling_policy"],
                split_manifest_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest(),
                runs=results,
            ),
        )
        write(
            OUT / "status.json",
            dict(
                state="training",
                completed=idx + 1,
                total=len(seeds),
                seed=seed,
                step=end,
                end=end,
            ),
        )
    write(
        OUT / "status.json",
        dict(state="complete", completed=len(results), total=len(seeds)),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        OUT.mkdir(parents=True, exist_ok=True)
        write(
            OUT / "failure.json", dict(error=repr(e), traceback=traceback.format_exc())
        )
        raise
