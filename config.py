r"""Kauldron config: fine-tune Gemma 3 1B on a JSON prompt/response dataset.

Expected JSON format (a list of objects):
  [
    {"prompt": "What is the capital of France?", "response": "Paris."},
    {"prompt": "Explain gravity.", "response": "Gravity is a force..."},
    ...
  ]

Train locally:
  python -m kauldron.main \
      --cfg=config.py \
      --cfg.workdir=/tmp/gemma-sft-workdir \
      --cfg.train_ds.data_source.path=/path/to/train.json

Train on Vertex AI (submit_job.py passes these flags automatically):
  python -m kauldron.main \
      --cfg=config.py \
      --cfg.workdir=gs://YOUR_BUCKET/workdir \
      --cfg.train_ds.data_source.path=gs://YOUR_BUCKET/data/train.json \
      --cfg.evals.test.ds.data_source.path=gs://YOUR_BUCKET/data/val.json
"""

from kauldron import konfig

with konfig.imports():
  from gemma import gm
  from kauldron import kd
  import optax


def get_config():
  batch_size = 8
  max_length = 512

  return kd.train.Trainer(
      seed=42,
      # ------------------------------------------------------------------ Data
      train_ds=_make_dataset(
          path="gs://YOUR_BUCKET/data/train.json",   # overridden via CLI
          training=True,
          batch_size=batch_size,
          max_length=max_length,
      ),
      # ----------------------------------------------------------------- Model
      model=gm.nn.Gemma3_1B(
          tokens="batch.input",
      ),
      # Load pretrained weights from Google's public GCS bucket
      init_transform=gm.ckpts.LoadCheckpoint(
          path=gm.ckpts.CheckpointPath.GEMMA3_1B_PT,
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
      optimizer=optax.adafactor(learning_rate=1e-3),
      checkpointer=kd.ckpts.Checkpointer(
          save_interval_steps=500,
      ),
      # ------------------------------------------------------------- Evaluation
      evals={
          "test": kd.evals.Evaluator(
              run=kd.evals.EveryNSteps(500),
              ds=_make_dataset(
                  path="gs://YOUR_BUCKET/data/val.json",   # overridden via CLI
                  training=False,
                  batch_size=batch_size,
                  max_length=max_length,
              ),
          ),
          "sampling": gm.evals.SamplerEvaluator(
              run=kd.evals.EveryNSteps(500),
              max_new_tokens=128,
              num_batches=1,
              ds=_make_dataset(
                  path="gs://YOUR_BUCKET/data/val.json",   # overridden via CLI
                  training=False,
                  sampling=True,
              ),
          ),
      },
  )


def _make_dataset(
    *,
    path: str,
    training: bool,
    sampling: bool = False,
    batch_size: int | None = None,
    max_length: int | None = None,
):
  tokenizer = gm.text.Gemma3Tokenizer()

  return kd.data.py.DataSource(
      data_source=kd.data.py.JsonDataSource(path=path),
      shuffle=training,
      num_epochs=None if training else 1,
      batch_size=None if sampling else batch_size,
      num_workers=4,
      transforms=[
          gm.data.Seq2SeqTask(
              # These must match the keys in your JSON objects
              in_prompt="prompt",
              in_response="response",
              # Output keys consumed by the model and loss
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
