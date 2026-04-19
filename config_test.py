"""Local test config using Gemma3 270M — small enough to run on CPU.

Run:
  python -m kauldron.main \
      --cfg=config_test.py \
      --cfg.workdir=/tmp/gemma-sft-test \
      --cfg.train_ds.path=data/sample.json \
      --cfg.evals.test.ds.path=data/sample.json \
      --cfg.evals.sampling.ds.path=data/sample.json
"""

from kauldron import konfig

with konfig.imports():
  from gemma import gm
  from kauldron import kd
  import optax


def get_config():
  batch_size = 2
  max_length = 128

  return kd.train.Trainer(
      seed=42,
      train_ds=_make_dataset(
          path="gs://YOUR_BUCKET/data/train.json",
          training=True,
          batch_size=batch_size,
          max_length=max_length,
      ),
      model=gm.nn.Gemma3_270M(
          tokens="batch.input",
      ),
      init_transform=gm.ckpts.LoadCheckpoint(
          path=gm.ckpts.CheckpointPath.GEMMA3_270M_PT,
      ),
      num_train_steps=2,
      train_losses={
          "xentropy": kd.losses.SoftmaxCrossEntropyWithIntLabels(
              logits="preds.logits",
              labels="batch.target",
              mask="batch.loss_mask",
          ),
      },
      optimizer=optax.adafactor(learning_rate=1e-3),
      checkpointer=kd.ckpts.Checkpointer(
          save_interval_steps=100,
      ),
      evals={},  # disabled for local CPU test to avoid OOM during eval
  )


def _make_dataset(
    *,
    path: str,
    training: bool,
    batch_size: int | None = None,
    max_length: int | None = None,
):
  tokenizer = gm.text.Gemma3Tokenizer()

  return kd.data.py.Json(
      path=path,
      shuffle=training,
      num_epochs=None if training else 1,
      batch_size=batch_size,
      num_workers=2,
      transforms=[
          gm.data.Seq2SeqTask(
              in_prompt="prompt",
              in_response="response",
              out_input="input",
              out_target="target",
              out_target_mask="loss_mask",
              tokenizer=tokenizer,
              max_length=max_length,
              truncate=True,
          ),
      ],
  )
