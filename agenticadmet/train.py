import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    PreTrainedTokenizer,
    Qwen2ForCausalLM
)

from trl import (
    GRPOConfig, 
    GRPOTrainer
)

from rl_dataset import load_polaris_dataset, validate_dataset
from loss import get_reward_functions
from callbacks.compute_metric import ComputeMetricsCallback
from trl.trainer.utils import selective_log_softmax 
from munch import Munch
import wandb

from trl import ModelConfig, get_peft_config
from datetime import datetime
import argparse
from typing import Any, Union

import torch.utils.data
from accelerate.utils import broadcast_object_list, gather, gather_object
from torch import nn


from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import pad, selective_log_softmax
from transformers import Trainer
import logging

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def get_tokenizer(
    model_args: ModelConfig, training_args, auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=False, # model_args.trust_remote_code
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template
    elif auto_set_chat_template and tokenizer.get_chat_template() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    return tokenizer


def get_model(model_name, attn_implementation="flash_attention_2", device="cuda:0"):
    # Initialize base model
    if attn_implementation is not None:
        kwargs_dict = {"attn_implementation": attn_implementation}
    else:
        kwargs_dict = {}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
        **kwargs_dict
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model

def get_dataset(params=["MLM", "HLM", "KSOL", "LogD", "MDR1-MDCKII"], subset_train=None, subset_valid=None, subset_test=None, rules_prompt_name="rules_v4", seed=42, properties=False, rewrite=False):
    dataset = load_polaris_dataset(params=params, rules_prompt_name=rules_prompt_name, seed=seed, properties=properties, rewrite=rewrite)

    print(f"Train set size: {len(dataset['train'])}")
    print(f"Test set size: {len(dataset['test'])}")

    if subset_train is not None:
        dataset["train"] = dataset["train"].select(range(subset_train))
    if subset_valid is not None:
        dataset["validation"] = dataset["validation"].select(range(subset_valid))
    if subset_test is not None:
        dataset["test"] = dataset["test"].select(range(subset_test))

    validate_dataset(dataset)
    return dataset


@dataclass
class GRPOScriptArguments:
    """
    Script arguments for GRPO training, specifically related to reward functions.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "compute_mae_v2"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'repetition_penalty'"        },
    )

    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-0.1,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )

class GRPOTrainer2(GRPOTrainer):

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1, use_cache=False).logits
        logits = logits[:, :-1, :]

        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        return loss
    
    
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        mae_not_reward = None
        rewards_len = len(self.reward_funcs) if not any("compute_mae_v2" in l.__name__ for l in self.reward_funcs) else len(self.reward_funcs)-1
        rewards_per_func = torch.zeros(len(prompts), rewards_len, device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if "compute_mae_v2" != reward_func.__name__:
                if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = Trainer._prepare_inputs(self, reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
            elif not self.model.training:
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                mae_not_reward = torch.tensor(output_reward_func, dtype=torch.float32, device=device)                

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        if mae_not_reward is not None and not self.model.training:
            mae_not_reward = gather(mae_not_reward).mean(0)
            self._metrics[f"rewards/MAE"].append(mae_not_reward.item())

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights[:rewards_len].to(device).unsqueeze(0)).sum(dim=1)
       
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            if "compute_mae_v2" not in reward_func_name:
                self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
            and wandb.run is not None
            and self.accelerator.is_main_process  # Ensure only main process logs
        ):
            import pandas as pd
            import os

            # Prepare DataFrame for logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            # Define artifact name and type
            artifact_name = f"completions_step_{self.state.global_step}"
            artifact_type = "dataset"

            # Save DataFrame to a CSV file
            os.makedirs(self.args.output_dir, exist_ok=True)  # Ensure the directory exists
            csv_path = os.path.join(self.args.output_dir, f"{artifact_name}.csv")
            df.to_csv(csv_path, index=False)

            # Create W&B Artifact
            artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
            artifact.add_file(csv_path)

            # Log the artifact
            wandb.run.log_artifact(artifact)

            print(f"Logged completions as W&B artifact: {artifact_name}")


        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
    
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        '--resume-from-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from (if any).'
    )
    parser.add_argument(
        '--eval-temperature',
        type=float,
        default=0.01,
        help='temerature'
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3.0e-06,
        help="LR"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int, 
        default=13*2,
        help="warmup steps"
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=8
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=6
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=13 # 6 is all the set, when we limit set to 100 samples
    )
    parser.add_argument(
        "--rules-prompt",
        type=str,
        default="rules_v4"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )
    parser.add_argument(
        "--reward-funcs",
        type=str,
        default="accuracy,compute_mae_v2",
        help="Comma-separated list of reward function names"
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=40,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup Ratio for Learning Rate"
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="linear",
        help="Learning Rate Scheduler Type"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight Decay"
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of generations."
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.25
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=6851+900+82+2000
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=6851+82+900
    )
    parser.add_argument(
        "--reward-weights",
        type=str,
        default=None,
        help="Comma-separated list od reward function weights"
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=2000, #1200,
        help="Max len of completion"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05
    )
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default=None,
        help="Comma-separated list of lora target modules"
    )
    parser.add_argument(
        "--subset-train",
        type=int,
        default=None,
        help="If you want to run on a subset of a train set."
    )
    parser.add_argument(
        "--subset-valid",
        type=int,
        default=None,
        help="If you want to run on a subset of the validation set."
    )
    parser.add_argument(
        "--vllm-device",
        type=str,
        default="cuda:0",
        help="String n the form cuda:0, id of device id to host vllm"
    )

    args = parser.parse_args()
    trainer_kwargs = {}
    if args.resume_from_checkpoint is not None:
        trainer_kwargs['resume_from_checkpoint'] = args.resume_from_checkpoint

    MODEL_NAME = args.model_name

    wandb.init(project="admet-challenge")
    wandb.config.update({"log_model": False})
    now = datetime.now()

    model_args_i = Munch.fromDict({
        "model_name_or_path": MODEL_NAME,
        "model_revision": "main",
        "trust_remote_code": False
        })
    training_args_i = Munch.fromDict({"chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"})
    
    tokenizer = get_tokenizer(model_args_i, training_args_i)
    model_args = ModelConfig(model_name_or_path=MODEL_NAME, use_peft=True, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, lora_target_modules=args.lora_target_modules)
    
    model_cuda = "cuda:0"
    model = get_model(MODEL_NAME, attn_implementation="flash_attention_2", device=model_cuda) #TODO: change to "flash_attention_2"
    
    if args.reward_weights is not None:
        reward_weights = [float(l) for l in args.reward_weights.split(",")]
    else:
        reward_weights = None

    seed = 42
    dataset = get_dataset(params=["LogD"], rules_prompt_name=args.rules_prompt, rewrite=False, properties=True, seed=seed, subset_train=args.subset_train, subset_valid=args.subset_valid) # TODO: change to default TODO: subset None 50 is 1/4 of the LogD dataset (200)
        
    text_table = wandb.Table(columns=["smiles_hash", "steps", "reward", "mae_median", "mae", "completion", "system_input", "user_prompt", "answer_parsed", "asnwer_val", "gold_val"])
    text_table_current = wandb.Table(columns=["smiles_hash", "steps", "reward", "mae_median", "mae", "completion", "system_input", "user_prompt", "answer_parsed", "asnwer_val", "gold_val"])
    reward_funcs_names = args.reward_funcs.split(",")
    script_args = GRPOScriptArguments(reward_funcs=reward_funcs_names)
    reward_functions = get_reward_functions(script_args, mae_thr=None, table=text_table, text_table_current=text_table_current) #TODO: check trl they had someshere gpro example and used different rewards including lenght reward
    
    training_args = TrainingArguments(
        output_dir=f"{os.environ.get('OUTPUT_DIR')}{now:%Y-%m-%d}/{now:%H-%M-%S}", #"./output",
        logging_dir="./logs/wandb/",
        num_train_epochs=args.num_train_epochs,             # Total number of training epochs
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        logging_first_step=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=False,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=seed,
        bf16=True,
        push_to_hub=False,
        report_to=["wandb"],
        run_name="test",
        disable_tqdm=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        do_eval=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type=args.lr_scheduler_type,
        max_steps=-1,
        eval_steps=args.eval_steps,
)

    grpo_config = GRPOConfig(
        **training_args.to_dict(), # Convert TrainingArguments to dictionary and unpack
        **{ 
        # REMOVED model_init_kwargs here 
        # We are passing the instantiated 'model' object, so GRPOTrainer doesn't need model_init_kwargs
        },
        num_generations=args.num_generations, #TODO: 16
        use_vllm=True, #TODO: use True
        vllm_device=args.vllm_device,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization, # TODO: 0.25 0.7
        vllm_max_model_len=args.vllm_max_model_len, #TODO: 2048
        max_prompt_length=args.max_prompt_length, #3024, #TODO: 800+
        max_completion_length=args.max_completion_length, #TODO: 1024+ (better 2048/4048 and more)
        temperature=args.eval_temperature, # TODO: temperature for math task
        reward_weights=reward_weights,
        log_completions=False,
        )

    grpo_trainer = GRPOTrainer2(
        model=model,                     
        reward_funcs=reward_functions, 
        args=grpo_config,              
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args)
    )
    
    print_trainable_parameters(grpo_trainer.model)

    try:
        train_result = grpo_trainer.train(**trainer_kwargs)
        wandb.log({"training_samples" : text_table})
    except KeyboardInterrupt:
        print("Training interrupted! Cleaning up...")
    finally:
        wandb.log({"training_samples": text_table})
        wandb.finish()

if __name__ == "__main__":
    main()