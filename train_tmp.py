# Import necessary libraries
import os
from dataclasses import dataclass, field
from typing import Optional

# Import PyTorch and Hugging Face Transformers
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    PreTrainedTokenizer,
    Qwen2ForCausalLM
)

# Import libraries from TRL (Transformers Reinforcement Learning)
from trl import (
    GRPOConfig, 
    GRPOTrainer
)

#source openr1/bin/activate
from dataset import load_polaris_dataset, validate_dataset
from loss import get_reward_functions
from callbacks.compute_metric import ComputeMetricsCallback
from trl.trainer.utils import selective_log_softmax 
from munch import Munch
import wandb
# wandb.login()

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

# import os
# os.environ["WANDB_CONSOLE"] = "wrap"

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def get_tokenizer(
    model_args: ModelConfig, training_args, auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    # https://github.com/huggingface/open-r1/blob/eeca246b078457bc0f69ba2e8297b799df0e2bda/src/open_r1/utils/model_utils.py#L11
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
        device_map=device, #TODO: how it affects the ddp https://huggingface.co/openai/whisper-large-v3/discussions/63
        low_cpu_mem_usage=True, #TODO: ??
        # use_safetensors=True, #TODO: ??
        **kwargs_dict
    )

    print(f"Model parameters: {model.num_parameters():,}")

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using device: {device}")

    # Move model to the appropriate device
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


# Define GRPOScriptArguments for reward function parameters
@dataclass
class GRPOScriptArguments:
    """
    Script arguments for GRPO training, specifically related to reward functions.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "compute_mae_v2"], #, "mae_not_reward"],
                                 #, "format", "reasoning_steps", "repetition_penalty"], #TODO: reasoning and repetition are mot the best, add thinking length reward
        metadata={
            # "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty'"
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

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        # print(input_ids[0].shape)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1, use_cache=False).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        # print(loss, advantages) 
        return loss
    
    # TODO: When updating trl version this may migrate to generate function
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        # smiles = [x["smiles"] for x in inputs] #TODO: debug
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
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
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

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

        # if (
        #     self.log_completions
        #     and self.state.global_step % self.args.logging_steps == 0
        #     and "wandb" in self.args.report_to
        # ):
        #     import pandas as pd

        #     # For logging
        #     table = {
        #         "step": [self.state.global_step] * len(rewards), #TODO: debug
        #         "prompt": gather_object(prompts_text),
        #         "completion": gather_object(completions_text),
        #         "reward": rewards.tolist(),
        #     }
        #     df = pd.DataFrame(table)

        #     # if wandb.run is not None and self.accelerator.is_main_process:
        #     #     wandb.log({"completions": wandb.Table(dataframe=df)})

        #     # if wandb.run is not None and self.accelerator.is_main_process:
        #     #     if "completions_table" not in wandb.run.summary:  # First-time initialization
        #     #         wandb.run.summary["completions_table"] = wandb.Table(columns=["step", "prompt", "completion", "reward"])
                
        #     #     table = wandb.run.summary["completions_table"]
                
        #     #     for _, row in df.iterrows():
        #     #         table.add_data(row["step"], row["prompt"], row["completion"], row["reward"])
                
        #     #     wandb.log({"completions": table})  # Log without overriding previous data

        #     if wandb.run is not None and self.accelerator.is_main_process:
        #         if not hasattr(wandb.run, "summary") or "completions_table" not in wandb.run.summary:
        #             wandb.run.summary["completions_table"] = wandb.Table(columns=["step", "prompt", "completion", "reward"])

        #         table = wandb.run.summary["completions_table"]

        #         for _, row in df.iterrows():
        #             table.add_data(row["step"], row["prompt"], row["completion"], row["reward"])

        #         wandb.log({"completions": table})

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
        default=2
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=2
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
        default=3 # 6 is all the set, when we limit set to 100 samples
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
        default=2,
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
        default=None
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--reward-weights",
        type=str,
        default="1,1",
        help="Comma-separated list od reward function weights"
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=512, #1200,
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
        default=10,
        help="If you want to run on a subset of a train set."
    )
    parser.add_argument(
        "--subset-valid",
        type=int,
        default=10,
        help="If you want to run on a subset of the validation set."
    )
    parser.add_argument(
        "--vllm-device",
        type=str,
        default="cuda:0",
        help="String n the form cuda:0, id of device id to host vllm"
    )
    #TODO: get default scheduler
    #TODO: get all global paths

    args = parser.parse_args()
    # Prepare keyword arguments for the trainer.
    trainer_kwargs = {}
    if args.resume_from_checkpoint is not None:
        trainer_kwargs['resume_from_checkpoint'] = args.resume_from_checkpoint

    # "./outputs/2025-02-26/20-13-13"
    # MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    MODEL_NAME = args.model_name # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" #DeepSeek-R1-Distill-Qwen-1.5B-GRPO
    MODEL_NAME = "Jiqing/tiny-random-qwen2"

    # https://github.com/huggingface/open-r1/blob/main/recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml
    # chat_template: "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"
    # dataset_name: open-r1/OpenR1-Math-220k
    # dataset_configs:
    # - default
    # system_prompt: "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"

    #TODO: reward function range

    wandb.init(project="admet-challenge", name="test")
    wandb.config.update({"log_model": False})
    now = datetime.now()

    model_args_i = Munch.fromDict({
        "model_name_or_path": MODEL_NAME,
        "model_revision": "main",
        "trust_remote_code": False # TODO: everyboudy sets to True and default is True
        })
    training_args_i = Munch.fromDict({"chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"})
    
    tokenizer = get_tokenizer(model_args_i, training_args_i)
    model_args = ModelConfig(model_name_or_path=MODEL_NAME, use_peft=True, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, lora_target_modules=args.lora_target_modules)
                             #, load_in_8bit=True) # TODO: if run it in a serverless sometimes loading of huggingface weights throws an error
    # TODO: we now use default lora setting, how do we choose the best configuration 
    # lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules=None, lora_modules_to_save=None, lora_task_type='CAUSAL_LM', use_rslora=False, load_in_8bit=False, load_in_4bit=False, bnb_4bit_quant_type='nf4', use_bnb_nested_quant=False

    print("!!!!! Model args", model_args)

    # TODO: get what following parameters people use
    # task_type=model_args.lora_task_type,
    #     r=model_args.lora_r,
    #     target_modules=model_args.lora_target_modules,
    #     lora_alpha=model_args.lora_alpha,
    #     lora_dropout=model_args.lora_dropout,
    #     bias="none",
    #     use_rslora=model_args.use_rslora,
    #     modules_to_save=model_args.lora_modules_to_save,
    
    #TODO: gpu utilization with falsh attention is 24, without 96 (now no difference)
    #TODO: (DONE) (+) with flash attention throuw warning that flash attention is attemted to be used in a model on cpu
    # You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with model.to('cuda')
    # https://huggingface.co/openai/whisper-large-v3/discussions/63 
    #TODO: falsh_attention no speedup
    model_cuda = "cuda:0"
    model = get_model(MODEL_NAME, attn_implementation="flash_attention_2", device=model_cuda) #TODO: change to "flash_attention_2"

    reward_weights = [float(l) for l in args.reward_weights.split(",")]

    print_trainable_parameters(model)
    # print("Model attention implementation: ", model.model.text_model._attn_implementation)
    print("Attention implementation:", model.config._attn_implementation)
    # for name, module in model.model.named_modules():
    #     if "attn" in name.lower() or "attention" in name.lower():
    #         print(name, "->", module.__class__)
    # dataset = get_dataset(params=["LogD"], rewrite=True, subset_train=50) # TODO: change to default TODO: subset None 50 is 1/4 of the LogD dataset (200)
    # subset_train=4, subset_valid=4, 
    seed = 42
    dataset = get_dataset(params=["LogD"], rules_prompt_name=args.rules_prompt, rewrite=False, properties=True, seed=seed, subset_train=args.subset_train, subset_valid=args.subset_valid) # TODO: change to default TODO: subset None 50 is 1/4 of the LogD dataset (200)
    print(len(dataset["train"]), len(dataset["validation"]), len(dataset["test"]))

    text_table = wandb.Table(columns=["smiles_hash", "steps", "reward", "mae_median", "mae", "completion", "system_input", "user_prompt", "answer_parsed", "asnwer_val", "gold_val"])
    text_table_current = wandb.Table(columns=["smiles_hash", "steps", "reward", "mae_median", "mae", "completion", "system_input", "user_prompt", "answer_parsed", "asnwer_val", "gold_val"])
    reward_funcs_names = args.reward_funcs.split(",")
    script_args = GRPOScriptArguments(reward_funcs=reward_funcs_names)
    reward_functions = get_reward_functions(script_args, mae_thr=0.5, table=text_table, text_table_current=text_table_current) #TODO: check trl they had someshere gpro example and used different rewards including lenght reward
    
    # "dirpath": f"{os.environ.get('AIP_MODEL_DIR', './outputs/')}{now:%Y-%m-%d}/{now:%H-%M-%S}"
    training_args = TrainingArguments(
        output_dir=f"{os.environ.get('OUTPUT_DIR')}{now:%Y-%m-%d}/{now:%H-%M-%S}", #"./output",
        logging_dir="./logs/wandb/",
        num_train_epochs=args.num_train_epochs,             # Total number of training epochs
        per_device_train_batch_size=args.per_device_train_batch_size,  # Batch size per device during training TODO: change to 16
        per_device_eval_batch_size=args.per_device_eval_batch_size,   # Batch size for evaluation TODO: why it says this   File "/home/alisavin/AgenticADMET/train.py", line 534, in <module>
        gradient_accumulation_steps=args.gradient_accumulation_steps, #4,  # Accumulate gradients to simulate larger batch size
        # learning_rate=1e-6,            # TODO: Initial learning rate for AdamW optimizer
        # learning_rate=2.0e-05, # took tuners https://github.com/Mryangkaitong/deepseek-r1-gsm8k/blob/main/recipes/DeepSeek-R1-Distill-Qwen-7B/grpo/config_demo.yaml
        learning_rate=args.lr, # took tuners https://github.com/Mryangkaitong/deepseek-r1-gsm8k/blob/main/recipes/DeepSeek-R1-Distill-Qwen-7B/grpo/config_demo.yaml
        warmup_ratio=args.warmup_ratio,              # Linear warmup over warmup_ratio fraction of training steps
        weight_decay=args.weight_decay,             # Apply weight decay to all layers except bias and LayerNorm weights
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,              # Log every X updates steps TODO: change based on number of steps
        logging_strategy="steps",
        logging_first_step=True,
        evaluation_strategy="steps",    # Evaluate every `eval_steps`
        save_strategy="epoch",      # Disables regular checkpoints
        save_total_limit=1,      # Makes sure no checkpoints are kept
        load_best_model_at_end=False,  # Disables saving the best model
        # save_steps=0,            # No saving at specific steps
        dataloader_num_workers=args.dataloader_num_workers,      # Number of subprocesses to use for data loading
        seed=seed,                       # Random seed for reproducibility
        bf16=True,                     # Use mixed precision BFP16 training #TODO: ??????
        push_to_hub=False,             # Whether to push the final model to Hugging Face Hub
        report_to=["wandb"],              # Reporting to no one
        run_name="test",
        disable_tqdm=False,
        gradient_checkpointing=True,   # Enable gradient checkpointing        
        remove_unused_columns=False,
        do_eval=False, #TODO: use
        gradient_checkpointing_kwargs={"use_reentrant": False}, # TODO: use
        lr_scheduler_type=args.lr_scheduler_type,
        # ---------
        # # TODO
        # # log_completions=True,
        # # log_level="info",
        # lr_scheduler_type="cosine_with_min_lr", #TODO: before trained with cosine with min lr
        # lr_scheduler_type="cosine", #TODO: tuners https://github.com/Mryangkaitong/deepseek-r1-gsm8k/blob/5dcf23a94b17dd970a142183c6bdbcadf1a75f47/recipes/DeepSeek-R1-Distill-Qwen-7B/grpo/config_demo.yaml#L39
        # lr_scheduler_kwargs={"min_lr_rate": 0.1}, #TODO: before used
        max_steps=-1, #TODO: change to -1
        eval_steps=args.eval_steps, #TODO: change to -1
        # log_level="debug"
)
    
    #TODO: reward, for each property set threashold for MAE to set to range 0 to 1
    #TODO: loss always 0

    grpo_config = GRPOConfig(
        **training_args.to_dict(), # Convert TrainingArguments to dictionary and unpack
        **{ 
        # REMOVED model_init_kwargs here 
        # We are passing the instantiated 'model' object, so GRPOTrainer doesn't need model_init_kwargs
        },
        num_generations=args.num_generations, #TODO: 16
        use_vllm=False, #TODO: use True
        vllm_device=args.vllm_device,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization, # TODO: 0.25 0.7
        vllm_max_model_len=args.vllm_max_model_len, #TODO: 2048
        max_prompt_length=args.max_prompt_length, #3024, #TODO: 800+
        max_completion_length=args.max_completion_length, #TODO: 1024+ (better 2048/4048 and more)
        temperature=args.eval_temperature, # TODO: temperature for math task
        reward_weights=reward_weights,
        log_completions=False
        )

    # for l in dataset['train']:
    #     print(len(l["prompt"][0]["content"])+len(l["prompt"][1]["content"]))

    # # TODO: does it help
    # peft_config = get_peft_config(model_args)
    # peft_config.label_names = ["prompt", "solution", "property", "smiles"]

    grpo_trainer = GRPOTrainer2(
        model=model,                      # Our initialized Qwen model
        reward_funcs=reward_functions,    # List of reward functions from previous step
        args=grpo_config,                # GRPOConfig (created from TrainingArguments)
        train_dataset=dataset['train'],   # Training dataset
        eval_dataset=dataset['validation'],    # Evaluation dataset
        # callbacks=callbacks              # List of callbacks
        processing_class=tokenizer, #TODO: check callback from config
        peft_config=get_peft_config(model_args), #TODO: check # label_names
        # callbacks=[ComputeMetricsCallback]  # Add callback
    )
    
    print_trainable_parameters(grpo_trainer.model)

    # Start the GRPO Training Loop
    train_result = grpo_trainer.train(**trainer_kwargs)
    wandb.log({"training_samples" : text_table})

    #TODO: LoRa isage produces error
    # pip install --upgrade --no-cache-dir --no-deps unsloth 

    #TODO: check if there is memory consumption bug
    #https://github.com/huggingface/trl/issues/2719
    # TODO: check if we can use deepspeed useing the accelerate script

    #TODO: no padding with whash attention, packing whould be True
    #https://github.com/huggingface/transformers/issues/28130

    #TODO: model supports multiple tasks: {'embed', 'reward', 'generate', 'score', 'classify'}. Defaulting to 'generate'.
    # WARNING 02-24 05:09:24 arg_utils.py:1145] The model has a long context length (131072). This may cause OOM errors during the initial memory profiling phase, or result in low performance due to small KV cache space. Consider setting --max-model-len to a smaller value.
    # INFO 02-24 05:09:24 llm_engine.py:234] 

    #TODO: save checkpoints
    
    #TODO: WARNING 02-24 18:05:56 arg_utils.py:1145] The model has a long context length (131072). This may cause OOM errors during the initial memory profiling phase, or result in low performance due to small KV cache space. Consider setting --max-model-len to a smaller value.
    # Memory Requirements: https://discuss.huggingface.co/t/llama-7b-gpu-memory-requirement/34323/8

    # TODO: Done (+) Why loss starts from 0 https://github.com/huggingface/open-r1/issues/239

if __name__ == "__main__":
    main()



# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)

# Lipinsky's rules define conditions under witch the molecule has the optimal absorbtion

# TODO: save generations

#TODO: ensure lora checkpoint is loaded

#TODO: check how "equations is deprecated, as it handled by the parser now" is thrown
#TODO: current prompt does not produce answer, but consistently puts final result in the last sentence inside \\boxed{}
#TODO: track generations at training 
# {Use computational tools like Dragon, LogE, ; ZINDAQ to calculate LogD using the given SMILES input.
# TODO: openr1 training recepy https://github.com/huggingface/open-r1/blob/main/src/open_r1/grpo.py
#TODO: log mae, check if it is the same as in a contest
#Whats with the resonings steps, do they grow, why not

#TODO: report median

#TODO: TODO: warmup from smaller learning rate

#TODO: !!! This does not support ddp

# /home/alisavin/AgenticADMET/openr1/lib/python3.11/site-packages/transformers/trainer.py:3423: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   torch.load(os.path.join(checkpoint, OPTIMIZER_NAME), map_location=map_location)
# Warning: The following arguments do not match the ones in the `trainer_state.json` within the checkpoint directory:
#         logging_steps: 1 (from args) != 10 (from trainer_state.json)
#   0%|                                                                                                              | 0/60 [00:00<?, ?it/s]/home/alisavin/AgenticADMET/openr1/lib/python3.11/site-packages/transformers/trainer.py:3119: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   checkpoint_rng_state = torch.load(rng_file)