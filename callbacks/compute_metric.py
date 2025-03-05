import hashlib
import numpy as np
from collections import defaultdict
from transformers import TrainerCallback
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse


class ComputeMetricsCallback(TrainerCallback):
    # def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

    def on_evaluate(self, args, state, control, **kwargs):
        """
        Custom callback to compute metrics after evaluation.
        """
        if "wandb" in args.report_to:
            import wandb
            text_table = wandb.Table(columns=["smiles_hash", "mse_median", "mse", "completion", "system_input", "user_prompt", "answer_parsed", "asnwer_val", "gold_val"])

        print(kwargs.get("metrics", {}))
        print(kwargs.keys())
        eval_dataloader = kwargs.get("eval_dataloader")
        if eval_dataloader is None:
            return

        print("Running custom metrics computation...")

        # Retrieve predictions, labels, and inputs
        predictions = state.predict(eval_dataloader).predictions
        labels = state.predict(eval_dataloader).label_ids

        # Extract SMILES inputs from dataset if available
        # inputs = getattr(trainer.eval_dataset, 'smiles', None)
        inputs_smiles = getattr(state.eval_dataset, 'smiles', None)
        inputs_prompts = getattr(state.eval_dataset, 'prompts', None)
        if inputs_smiles is None:
            raise ValueError("This metrics function requires inputs that contain SMILES strings")

        # Process predictions
        contents = [str(pred) for pred in predictions]

        # Group by SMILES
        smiles2conts = defaultdict(list)
        for content, gold_val, smiles_i, inputs_prompts_i in zip(contents, labels, inputs_smiles, inputs_prompts):
            answer_val = None
            mse = 4
            if gold_val is not None:
                # Use your parsing function to extract the answer
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )

                if len(answer_parsed) > 0 and not isinstance(answer_parsed[0], str):
                    answer_val = float(answer_parsed[0])
                    mse = (answer_val - float(gold_val))**2
            smiles_hash = hashlib.blake2b(smiles_i.encode('utf-8'), digest_size=4).hexdigest()
            smiles2conts[smiles_hash].append({
                "completion": content,
                "answer_parsed": answer_parsed,
                "answer_val": answer_val,
                "gold_val": gold_val,
                "mse": mse,
                "system_input": inputs_prompts_i[0]["content"],
                "user_prompt": inputs_prompts_i[1]["content"],
            })

        # Calculate metrics
        median_maes = []
        for k, v in smiles2conts.items():
            answers_g = [v_i["answer_val"] for v_i in v]
            valid_answers = [float(v_i) for v_i in answers_g if v_i is not None]

            if valid_answers:
                answer_median = np.median(valid_answers)
                gold_val = float(v[0]["gold_val"]) if v[0]["gold_val"] is not None else None
                if gold_val is not None:
                    mse_median = (gold_val - answer_median)**2
                    median_maes.append(mse_median)
                    for v_i in v:
                        text_table.add_data(k, mse_median, v_i["mse"], v_i["completion"], v_i["system_input"], v_i["user_prompt"], v_i["answer_parsed"], v_i["answer_val"], v_i["gold_val"]) 


        # Compute final metric
        median_mae = np.mean(median_maes) if median_maes else float(4.0)

        # Log results
        if "wandb" in args.report_to:
            print("Report!!!!!!!!")
            wandb.log({"eval/median_mae": median_mae}, step=state.global_step)
            wandb.log({"training_samples" : text_table})

        print(f"Computed median MAE: {median_mae}")