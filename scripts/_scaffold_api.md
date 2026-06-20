# API reference for experiment scaffolds (devsys)

Run python via `.venv/bin/python`. Package import root: `devsys`.

## Doctrine contract (REQUIRED)
Subclass `devsys.experiments.base.Experiment`. Set class attrs (all non-empty or the class
cannot be defined): `id` (str, MUST equal the config filename stem), `metric` (tuple[str]),
`baseline` (str), `ablation` (str), `null_hypothesis` (str), `tier` (one of:
"cpu-now","gpu-later","env-later","2.1-only"). Implement:
    def run(self, cfg, device, run_dir) -> dict   # json-serializable metrics, includes the null check
`cfg` is an OmegaConf DictConfig: read your params at `cfg.experiment.*`; also `cfg.shell.*`,
`cfg.encoder.*`, `cfg.device.*`, `cfg.seed`. `device` is a DeviceInfo (`device.device` is a
torch.device). `run_dir` is a Path you may write plots to (mkdir parents first).

## devices
`from devsys.devices import safe_to, resolve` ; `safe_to(tensor, device.device)`.

## substrate.datasets
`from devsys.substrate.datasets import make_task_stream, noisy_tv_dataset, Task`
- make_task_stream(n_tasks, dim, classes_per_task, samples_per_task, separation,
  incremental="class|task|domain", forward_dynamics=False, seed=0) -> list[Task]
- Task: .x [N,D] float, .y [N] long, .xnext [N,D] or None, .n_classes int, .task_id int
- noisy_tv_dataset(dim, n, separation, noise_scale, seed) -> {"learnable":Task,"noise":Task}
  (learnable has predictable .xnext; noise has irreducibly random .xnext)

## learning
`from devsys.learning.backprop import Learner, TrainConfig`
- TrainConfig(epochs_per_task, batch_size, replay_batch, base_lr, adapt_threshold)
- Learner(model, device, train_cfg, buffer=None, consolidation=None, plasticity=None,
  neuromod=None, seed=0); .train_task(task, progress0, progress1)->int(steps);
  .evaluate(list[Task])->list[float acc]
`from devsys.learning.alternatives import RULES`  # {name: train_fn}; train_fn(x,y,hidden,epochs,lr,seed)->RuleResult
  RuleResult has .test_acc .train_acc .seconds .local .weight_transport .separate_backward .activation_memory .chance

## shell
`from devsys.shell import ReplayBuffer, Consolidation, PlasticityController, Neuromodulation, Ensemble, Predictor, ClassHead, GaussianHead`
- ReplayBuffer(capacity, dim, key_dim=None, prioritized=True, alpha=.6, beta=.4, index="brute", eviction="reservoir", seed=0)
  .add(x,y,key=None,priority=None) .sample(batch)->{"x","y","idx","is_weight"} .retrieve(q,k)->{"idx","x","y","dist"}
- Consolidation(cfg) where cfg has .method("none|ewc|si|both"), .ewc_lambda, .si_c, .si_xi, .fisher_samples; .penalty(model); .begin_task(model); .after_step(model); .consolidate(model, batches, loss_fn)
- PlasticityController(cfg) cfg has .schedule("hard|soft|learned"), .lr, .rigidity, .pnn_fraction; .lr_scale(progress, signal=0.0)->float; .lr(progress,signal); .init_pnn(model); .update_rigidity(model); .rigidity_penalty(model); .apply_pnn_freeze(model)
- Neuromodulation(cfg) cfg has .enabled,.surprise_gain,.novelty_gain,.uncertainty_gain,.gate_floor,.gate_ceil; .signals(pred,target,disagreement=None,novelty=None)->dict; .gate(name,value)->float
- Ensemble(make_member_callable, size, bootstrap=False); .mean_and_disagreement(x)->(mean,[B]disagreement)
- Predictor(dim, hidden, depth, dropout=0, action_dim=0, layernorm=True)  (latent->latent)
- ClassHead(dim, n_classes, hidden=512, depth=1); GaussianHead(dim, out, hidden, depth)
You may build cfg-like blocks with `from omegaconf import OmegaConf; OmegaConf.create({...})`.
You may also read the shared shell config off `cfg.shell.buffer`, `cfg.shell.consolidation`, etc.

## metrics
`from devsys.metrics import ContinualResult, FrontierPoint, frontier_auc, retention_from_bwt`
- ContinualResult(R=list[list[float]], chance=float, adapt_steps=list[int]); .backward_transfer(), .forward_transfer(), .avg_accuracy(), .adaptation_speed(), .summary()
- FrontierPoint(name, adaptation, retention); frontier_auc(list[FrontierPoint])->float; retention_from_bwt(bwt)->float

## diagnostics
`from devsys.diagnostics import linear_probe, noisy_tv_diagnostic, fisher_trace_over_training, critical_period_signature, reliability`

## encoder (E6 dense vs pooled)
`from devsys.substrate.encoder import load_encoder` ; `from omegaconf import OmegaConf`
load each encoder config yaml (configs/encoder/<name>.yaml) via OmegaConf.load, pass to load_encoder -> FrozenEncoder.
.encode(clips[B,C,T,H,W])-> pooled [B,D] or dense [B,N,D]; .spec.embed_dim, .spec.dense. Falls back to frozen-random (no weights), which is fine.

## Config yaml (configs/experiment/<id>.yaml)
Must include: id (==filename stem), name, module, metric (list), null_hypothesis (str), tier, plus your params.
NOTE: use key `null_hypothesis`, NEVER `null` (yaml parses `null` as None and breaks OmegaConf).

## Form rules
BLACKHOLE: dense, flat, few load-bearing lines, no ceremony, comments only where they carry signal.
NO em dashes or en dashes anywhere (use commas/colons/parentheses). Keep run() fast (toy scale, seconds).
Do NOT edit __init__.py or scaffolds.py (registration is wired separately). Do NOT add dependencies.
