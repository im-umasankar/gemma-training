r"""Kauldron config: fine-tune Gemma 4 E2B on TheFinAI/Fino1_Reasoning_Path_FinQA.

Inspired by: https://www.datacamp.com/tutorial/fine-tune-gemma-3
Dataset:     TheFinAI/Fino1_Reasoning_Path_FinQA (500 financial reasoning examples)
Model:       Gemma 4 E2B (effective 2B — smallest available Gemma 4 model)

Note: There is no Gemma 4 1B. The Gemma 4 family starts at E2B (~2B params).

Dataset fields:
  - "Open-ended Verifiable Question" → used as the prompt
  - "Complex_CoT"                   → chain-of-thought reasoning (appended to prompt)
  - "Response"                      → target response

Prompt format (matches DataCamp tutorial style):
  <start_of_turn>user
  Question: {question}

  Think step by step:
  {chain_of_thought}<end_of_turn>
  <start_of_turn>model

Train locally (2-step smoke test):
  python -m kauldron.main \
      --cfg=config_gemma4.py \
      --cfg.workdir=/tmp/gemma4-finqa \
      --cfg.num_train_steps=2

Train on Vertex AI (via submit_job.py):
  python submit_job.py \
      --project gemma-training-winfiny \
      --region us-central1 \
      --bucket gs://gemma-training-winfiny-data \
      --train_data gs://gemma-training-winfiny-data/data/finqa_train.json \
      --val_data   gs://gemma-training-winfiny-data/data/finqa_val.json \
      --config config_gemma4.py \
      --skip_build
"""

from __future__ import annotations

from kauldron import konfig

# FinQAFormat and gemma/kd/optax are all imported inside konfig.imports()
# so Kauldron treats them as konfig-registered configurables.
with konfig.imports():
  from gemma import gm
  from kauldron import kd
  import optax
  import finqa_transform  # registers FinQAFormat with the konfig system


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_config():
  batch_size = 4
  max_length = 512

  return kd.train.Trainer(
      seed=42,
      # ------------------------------------------------------------------ Data
      train_ds=_make_dataset(
          split="train",
          training=True,
          batch_size=batch_size,
          max_length=max_length,
      ),
      # ----------------------------------------------------------------- Model
      model=gm.nn.Gemma4_E2B(
          tokens="batch.input",
      ),
      init_transform=gm.ckpts.LoadCheckpoint(
          path=gm.ckpts.CheckpointPath.GEMMA4_E2B_PT,
      ),
      # --------------------------------------------------------------- Training
      num_train_steps=5_000,
      train_losses={
          "xentropy": kd.losses.SoftmaxCrossEntropyWithIntLabels(
              logits="preds.logits",
              labels="batch.target",
              mask="batch.loss_mask",
          ),
      },
      optimizer=optax.adafactor(learning_rate=2e-4),
      checkpointer=kd.ckpts.Checkpointer(
          save_interval_steps=500,
      ),
      # ------------------------------------------------------------- Evaluation
      evals={
          "test": kd.evals.Evaluator(
              run=kd.evals.EveryNSteps(500),
              ds=_make_dataset(
                  split="train",   # dataset has no official val split — use train as proxy
                  training=False,
                  batch_size=batch_size,
                  max_length=max_length,
              ),
          ),
          "sampling": gm.evals.SamplerEvaluator(
              run=kd.evals.EveryNSteps(500),
              max_new_tokens=256,
              num_batches=1,
              ds=_make_dataset(
                  split="train",
                  training=False,
                  sampling=True,
              ),
          ),
      },
  )


def _make_dataset(
    *,
    split: str,
    training: bool,
    sampling: bool = False,
    batch_size: int | None = None,
    max_length: int | None = None,
):
  tokenizer = gm.text.Gemma3Tokenizer()  # Gemma 4 uses the same tokenizer API

  return kd.data.py.HuggingFace(
      path="TheFinAI/Fino1_Reasoning_Path_FinQA",
      split=split,
      shuffle=training,
      num_epochs=None if training else 1,
      batch_size=None if sampling else batch_size,
      num_workers=4,
      transforms=[
          # Step 1: format raw dataset fields into prompt/response
          finqa_transform.FinQAFormat(),
          # Step 2: tokenize and create loss mask (only response tokens)
          gm.data.Seq2SeqTask(
              in_prompt="prompt",
              in_response="response",
              out_input="input",
              out_target="target",
              out_target_mask="loss_mask",
              tokenizer=tokenizer,
              max_length=None if sampling else max_length,
              truncate=True,
              sampling=sampling,
          ),
      ],
  )
