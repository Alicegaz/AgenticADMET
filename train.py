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
    Trainer
)

# Import libraries from TRL (Transformers Reinforcement Learning)
from trl import (
    GRPOConfig, 
    GRPOTrainer
)
from trl.trainer.grpo_trainer import maybe_apply_chat_template, pad, gather_object, broadcast_object_list, is_conversational, apply_chat_template, unwrap_model_for_generation, gather, Union, Any, nn

#source openr1/bin/activate
from dataset import load_polaris_dataset, validate_dataset
from loss import get_reward_functions
import logging
from trl.trainer.utils import selective_log_softmax 
from munch import Munch
import wandb
wandb.login()

from transformers import AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def get_tokenizer(
    model_args: ModelConfig, training_args, auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    # https://github.com/huggingface/open-r1/blob/eeca246b078457bc0f69ba2e8297b799df0e2bda/src/open_r1/utils/model_utils.py#L11
    print("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=False, # model_args.trust_remote_code
    )
    print("tokenizer loaded")

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template
    elif auto_set_chat_template and tokenizer.get_chat_template() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    print("chat template")
    # if processing_class is None:
    #     processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
    return tokenizer

# def get_tokenizer(model_name):
#     # Initialize tokenizer with chat template
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name,
#         trust_remote_code=True,
#         padding_side="right"
#     )

#     # Set pad token if not set
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     print(f"Vocabulary size: {len(tokenizer)}")
#     print(f"Model max length: {tokenizer.model_max_length}")
#     print(f"Pad token: {tokenizer.pad_token}")
#     print(f"EOS token: {tokenizer.eos_token}")

#     return tokenizer

def get_model(model_name, attn_implementation="flash_attention_2"):
    # Initialize base model
    if attn_implementation is not None:
        kwargs_dict = {"attn_implementation": attn_implementation}
    else:
        kwargs_dict = {}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
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

def get_dataset():
    dataset = load_polaris_dataset()

    print(f"Train set size: {len(dataset['train'])}")
    print(f"Test set size: {len(dataset['test'])}")

    validate_dataset(dataset)
    return dataset


# Define GRPOScriptArguments for reward function parameters
@dataclass
class GRPOScriptArguments:
    """
    Script arguments for GRPO training, specifically related to reward functions.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "reasoning_steps", "repetition_penalty"], #TODO: reasoning and repetition are mot the best, add thinking length reward
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

@dataclass
class ModelConfig:
    """
    Configuration for the model.
    """

    # model_name_or_path: str = field(
    #     default=MODEL_NAME, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    # )
    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_revision: Optional[str] = field(
        default="main", metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16", metadata={"help": "Override the default `torch_dtype` and load the model under this dtype."}
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Trust remote code when loading model and tokenizer."}
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2", metadata={"help": "Attention implementation to use. 'flash_attention_2' or None"}
    )

class GRPOTrainer2(GRPOTrainer):

    # def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
    #     # Extract additional columns we want to pass to reward function 
    #     additional_inputs = {
    #         key: [example[key] for example in inputs] 
    #         for key in ["solution", "ground_truth"] 
    #         if key in inputs[0]
    #     }
        
    #     # Call parent's _prepare_inputs 
    #     prepared = super()._prepare_inputs(inputs)
        
    #     # Add our additional inputs to the prepared dict
    #     if additional_inputs:
    #         prepared.update(additional_inputs)
            
    #     return prepared

    # def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
    #     device = self.accelerator.device
    #     prompts = [x["prompt"] for x in inputs]
    #     prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
    #     prompt_inputs = self.processing_class(
    #         prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
    #     )
    #     prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
    #     prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    #     if self.max_prompt_length is not None:
    #         prompt_ids = prompt_ids[:, -self.max_prompt_length :]
    #         prompt_mask = prompt_mask[:, -self.max_prompt_length :]

    #     # Generate completions using either vLLM or regular generation
    #     if self.args.use_vllm:
    #         # First, have main process load weights if needed
    #         if self.state.global_step != self._last_loaded_step:
    #             self._move_model_to_vllm()
    #             self._last_loaded_step = self.state.global_step

    #         # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
    #         all_prompts_text = gather_object(prompts_text)
    #         if self.accelerator.is_main_process:
    #             outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
    #             completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
    #         else:
    #             completion_ids = [None] * len(all_prompts_text)
    #         # Broadcast the completions from the main process to all processes, ensuring each process receives its
    #         # corresponding slice.
    #         completion_ids = broadcast_object_list(completion_ids, from_process=0)
    #         process_slice = slice(
    #             self.accelerator.process_index * len(prompts),
    #             (self.accelerator.process_index + 1) * len(prompts),
    #         )
    #         completion_ids = completion_ids[process_slice]

    #         # Pad the completions, and concatenate them with the prompts
    #         completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
    #         # completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id, padding_side="left")
    #         completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
    #         prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    #     else:
    #         # Regular generation path
    #         with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
    #             prompt_completion_ids = unwrapped_model.generate(
    #                 prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
    #             )

    #         # Compute prompt length and extract completion ids
    #         prompt_length = prompt_ids.size(1)
    #         prompt_ids = prompt_completion_ids[:, :prompt_length]
    #         completion_ids = prompt_completion_ids[:, prompt_length:]

    #     # Mask everything after the first EOS token
    #     is_eos = completion_ids == self.processing_class.eos_token_id
    #     eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    #     eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    #     sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    #     completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    #     # Concatenate prompt_mask with completion_mask for logit computation
    #     attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

    #     logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

    #     with torch.inference_mode():
    #         if self.ref_model is not None:
    #             ref_per_token_logps = self._get_per_token_logps(
    #                 self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
    #             )
    #         else:
    #             with self.accelerator.unwrap_model(self.model).disable_adapter():
    #                 ref_per_token_logps = self._get_per_token_logps(
    #                     self.model, prompt_completion_ids, attention_mask, logits_to_keep
    #                 )

    #     # Decode the generated completions
    #     completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
    #     if is_conversational(inputs[0]):
    #         completions = []
    #         for prompt, completion in zip(prompts, completions_text):
    #             bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
    #             completions.append([{"role": "assistant", "content": bootstrap + completion}])
    #     else:
    #         completions = completions_text

    #     rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
    #     for i, (reward_func, reward_processing_class) in enumerate(
    #         zip(self.reward_funcs, self.reward_processing_classes)
    #     ):
    #         if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
    #             if is_conversational(inputs[0]):
    #                 messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
    #                 texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
    #             else:
    #                 texts = [p + c for p, c in zip(prompts, completions)]
    #             reward_inputs = reward_processing_class(
    #                 texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
    #             )
    #             reward_inputs = super()._prepare_inputs(reward_inputs)
    #             with torch.inference_mode():
    #                 rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
    #         else:
    #             # Repeat all input columns (but "prompt" and "completion") to match the number of generations
    #             keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
    #             reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
    #             output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
    #             rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

    #     # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
    #     # completions may be distributed across processes
    #     rewards_per_func = gather(rewards_per_func)

    #     # Apply weights to each reward function's output and sum
    #     rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

    #     # Compute grouped-wise rewards
    #     mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
    #     std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

    #     # Normalize the rewards to compute the advantages
    #     mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    #     std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    #     advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    #     # Slice to keep only the local part of the data
    #     process_slice = slice(
    #         self.accelerator.process_index * len(prompts),
    #         (self.accelerator.process_index + 1) * len(prompts),
    #     )
    #     advantages = advantages[process_slice]

    #     # Log the metrics
    #     reward_per_func = rewards_per_func.mean(0)
    #     for i, reward_func in enumerate(self.reward_funcs):
    #         if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
    #             reward_func_name = reward_func.config._name_or_path.split("/")[-1]
    #         else:
    #             reward_func_name = reward_func.__name__
    #         self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

    #     self._metrics["reward"].append(rewards.mean().item())
    #     self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

    #     if (
    #         self.log_completions
    #         and self.state.global_step % self.args.logging_steps == 0
    #         and "wandb" in self.args.report_to
    #     ):
    #         import pandas as pd

    #         # For logging
    #         table = {
    #             "step": [str(self.state.global_step)] * len(rewards),
    #             "prompt": gather_object(prompts_text),
    #             "completion": gather_object(completions_text),
    #             "reward": rewards.tolist(),
    #         }
    #         df = pd.DataFrame(table)

    #         if wandb.run is not None and self.accelerator.is_main_process:
    #             wandb.log({"completions": wandb.Table(dataframe=df)})

    #     return {
    #         "prompt_ids": prompt_ids,
    #         "prompt_mask": prompt_mask,
    #         "completion_ids": completion_ids,
    #         "completion_mask": completion_mask,
    #         "ref_per_token_logps": ref_per_token_logps,
    #         "advantages": advantages,
    #     }
    
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


def main():
    # MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    # MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" #DeepSeek-R1-Distill-Qwen-1.5B-GRPO
    # MODEL_NAME = "nickypro/tinyllama-15M"
    OUTPUT_DIR = "data/Qwen-GRPO-training" # For saving our trained model

    # https://github.com/huggingface/open-r1/blob/main/recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml
    # chat_template: "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"
    # dataset_name: open-r1/OpenR1-Math-220k
    # dataset_configs:
    # - default
    # system_prompt: "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"

    #TODO: reward function range

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize tokenizer with chat template
    # tokenizer = get_tokenizer(MODEL_NAME)

    model_args_i = Munch.fromDict({
        "model_name_or_path": MODEL_NAME,
        "model_revision": "main",
        "trust_remote_code": False
        })
    training_args_i = Munch.fromDict({"chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"})
    
    tokenizer = get_tokenizer(model_args_i, training_args_i)
    model_args = ModelConfig(model_name_or_path=MODEL_NAME)
    
    #TODO: gpu utilization with falsh attention is 24, without 96
    #TODO: with flash attention throuw warning that flash attention is attemted to be used in a model on cpu
    model = get_model(MODEL_NAME, attn_implementation=None) #TODO: change to "flash_attention_2"
    dataset = get_dataset()

    script_args = GRPOScriptArguments()
    reward_functions = get_reward_functions(script_args) #TODO: check trl they had someshere gpro example and used different rewards including lenght reward
    print(reward_functions)

    training_args = TrainingArguments(
        output_dir="./output",
        logging_dir="./logs/wandb/",
        num_train_epochs=1,             # Total number of training epochs
        per_device_train_batch_size=4,  # Batch size per device during training
        per_device_eval_batch_size=4,   # Batch size for evaluation TODO: why it says this   File "/home/alisavin/AgenticADMET/train.py", line 534, in <module>
#     main()
#   File "/home/alisavin/AgenticADMET/train.py", line 519, in main
#     grpo_trainer = GRPOTrainer2(
#                    ^^^^^^^^^^^^^
#   File "/home/alisavin/AgenticADMET/openr1/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py", line 346, in __init__
#     raise ValueError(
# ValueError: The global train batch size (1 x 8) must be evenly divisible by the number of generations per prompt (16). Given the current train batch size, the valid values for the number of generations are: [2, 4, 8].
# Traceback (most recent call last):
#   File "/home/alisavin/AgenticADMET/train.py", line 534, in <module>
#     main()
#   File "/home/alisavin/AgenticADMET/train.py", line 519, in main
#     grpo_trainer = GRPOTrainer2(
#                    ^^^^^^^^^^^^^
#   File "/home/alisavin/AgenticADMET/openr1/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py", line 346, in __init__
#     raise ValueError(
# ValueError: The global train batch size (1 x 8) must be evenly divisible by the number of generations per prompt (16). Given the current train batch size, the valid values for the number of generations are: [2, 4, 8].
        gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
        learning_rate=1e-6,            # Initial learning rate for AdamW optimizer
        warmup_ratio=0.1,              # Linear warmup over warmup_ratio fraction of training steps
        weight_decay=0.01,             # Apply weight decay to all layers except bias and LayerNorm weights
        logging_steps=1,              # Log every X updates steps
        logging_strategy="steps",
        logging_first_step=True,
        evaluation_strategy="epoch",    # Evaluate every `eval_steps`
        save_strategy="no",      # Disables regular checkpoints
        save_total_limit=0,      # Makes sure no checkpoints are kept
        load_best_model_at_end=False,  # Disables saving the best model
        save_steps=0,            # No saving at specific steps
        dataloader_num_workers=4,      # Number of subprocesses to use for data loading
        seed=42,                       # Random seed for reproducibility
        bf16=True,                     # Use mixed precision BFP16 training #TODO: ??????
        push_to_hub=False,             # Whether to push the final model to Hugging Face Hub
        report_to=["wandb"],              # Reporting to no one
        run_name="test",
        disable_tqdm=False,
        gradient_checkpointing=True,   # Enable gradient checkpointing        
        remove_unused_columns=False,
        do_eval=False, #TODO: use
        gradient_checkpointing_kwargs={"use_reentrant": False}, # TODO: use
        # ---------
        # # TODO
        # # log_completions=True,
        # # log_level="info",
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        max_steps=-1, #TODO: change to -1
        # - 1.0
        # - 1.0
        # push_to_hub=False,             # Whether to push the final model to Hugging Face Hub
    )
    
    #TODO: reward, for each property set threashold for MAE to set to range 0 to 1
    #TODO: loss always 0

    # Create GRPOConfig from TrainingArguments
    grpo_config = GRPOConfig(
        **training_args.to_dict(), # Convert TrainingArguments to dictionary and unpack
        **{ 
        # REMOVED model_init_kwargs here 
        # We are passing the instantiated 'model' object, so GRPOTrainer doesn't need model_init_kwargs
        },
        num_generations=4, #TODO: 16
        use_vllm=True, #TODO: use True
        vllm_device="auto",
        vllm_gpu_memory_utilization=0.5, # 0.7
        max_prompt_length=800, #TODO: 800+
        max_completion_length=1024, #TODO: 1024+ (better 2048/4048 and more)
        temperature=0.7,
        reward_weights=[1.0, 1.0, 0.5, 0.5]
        )

    # for l in dataset['train']:
    #     print(len(l["prompt"][0]["content"])+len(l["prompt"][1]["content"]))

    grpo_trainer = GRPOTrainer2(
        model=model,                      # Our initialized Qwen model
        reward_funcs=reward_functions,    # List of reward functions from previous step
        args=grpo_config,                # GRPOConfig (created from TrainingArguments)
        train_dataset=dataset['train'],   # Training dataset
        eval_dataset=dataset['test'],    # Evaluation dataset
        # callbacks=callbacks              # List of callbacks
        processing_class=tokenizer #TODO: check callback from config
    )

    # Start the GRPO Training Loop
    train_result = grpo_trainer.train()


if __name__ == "__main__":
    main()
