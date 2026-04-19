"""Local CPU trial config for config_gemma4.py pipeline validation.

Uses Gemma 3 270M (fits in CPU RAM) with the same FinQA data pipeline
as config_gemma4.py to validate FinQAFormat transform and HuggingFace loading.

Run:
  python -m kauldron.main \
      --cfg=config_gemma4_test.py \
      --cfg.workdir=/tmp/gemma4-finqa-test \
      --cfg.num_train_steps=2
"""

from __future__ import annotations

from kauldron import konfig

with konfig.imports():
  from gemma import gm
  from kauldron import kd
  import optax
  import finqa_transform  # registers FinQAFormat with konfig


def get_config():
  batch_size = 2
  max_length = 256

  return kd.train.Trainer(
      seed=42,
      train_ds=_make_dataset(batch_size=batch_size, max_length=max_length),
      model=gm.nn.Gemma3_270M(tokens="batch.input"),
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
      optimizer=optax.adafactor(learning_rate=2e-4),
      checkpointer=kd.ckpts.Checkpointer(save_interval_steps=100),
      evals={},  # disabled to avoid OOM on CPU
  )


def _make_dataset(*, batch_size: int, max_length: int):
  tokenizer = gm.text.Gemma3Tokenizer()

  return kd.data.py.HuggingFace(
      path="TheFinAI/Fino1_Reasoning_Path_FinQA",
      split="train",
      shuffle=True,
      num_epochs=None,
      batch_size=batch_size,
      num_workers=2,
      transforms=[
          finqa_transform.FinQAFormat(),
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
