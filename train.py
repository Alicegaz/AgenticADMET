# Import necessary libraries
import os
from dataclasses import dataclass, field
from typing import Optional

# Import PyTorch and Hugging Face Transformers
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)

# Import libraries from TRL (Transformers Reinforcement Learning)
from trl import (
    GRPOConfig, 
    GRPOTrainer
)

#source openr1/bin/activate
from dataset import load_polaris_dataset, validate_dataset
from loss import get_reward_functions
from trl.trainer.utils import selective_log_softmax 
from munch import Munch
import wandb
wandb.login()

from transformers import AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_peft_config
import wandb
from datetime import datetime


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
        device_map="cuda:0", #TODO: how it affects the ddp https://huggingface.co/openai/whisper-large-v3/discussions/63
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

def get_dataset(params=["MLM", "HLM", "KSOL", "LogD", "MDR1-MDCKII"], subset_train=None, subset_valid=None, subset_test=None, rewrite=False):
    dataset = load_polaris_dataset(params=params, rewrite=rewrite)

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
    # MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" #DeepSeek-R1-Distill-Qwen-1.5B-GRPO
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

    wandb.init(project="admet-challenge")
    wandb.config.update({"log_model": False, "run_name": "test"})
    now = datetime.now()

    # Initialize tokenizer with chat template
    # tokenizer = get_tokenizer(MODEL_NAME)

    model_args_i = Munch.fromDict({
        "model_name_or_path": MODEL_NAME,
        "model_revision": "main",
        "trust_remote_code": False # TODO: everyboudy sets to True and default is True
        })
    training_args_i = Munch.fromDict({"chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"})
    
    tokenizer = get_tokenizer(model_args_i, training_args_i)
    model_args = ModelConfig(model_name_or_path=MODEL_NAME, use_peft=True, load_in_8bit=True) # TODO: if run it in a serverless sometimes loading of huggingface weights throws an error
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

    # model = get_model(MODEL_NAME, attn_implementation="flash_attention_2") #TODO: change to "flash_attention_2"
    model = get_model(MODEL_NAME, attn_implementation="flash_attention_2") #TODO: change to "flash_attention_2"
    print_trainable_parameters(model)
    # print("Model attention implementation: ", model.model.text_model._attn_implementation)
    print("Attention implementation:", model.config._attn_implementation)
    # for name, module in model.model.named_modules():
    #     if "attn" in name.lower() or "attention" in name.lower():
    #         print(name, "->", module.__class__)
    dataset = get_dataset(params=["LogD"], rewrite=False, subset_train=50) # TODO: change to default TODO: subset None 50 is 1/4 of the LogD dataset (200)
    print(len(dataset["train"]), len(dataset["validation"]), len(dataset["test"]))

    script_args = GRPOScriptArguments()
    reward_functions = get_reward_functions(script_args) #TODO: check trl they had someshere gpro example and used different rewards including lenght reward
    
    # "dirpath": f"{os.environ.get('AIP_MODEL_DIR', './outputs/')}{now:%Y-%m-%d}/{now:%H-%M-%S}"
    training_args = TrainingArguments(
        output_dir=f"{os.environ.get('AIP_MODEL_DIR', './outputs/')}{now:%Y-%m-%d}/{now:%H-%M-%S}", #"./output",
        logging_dir="./logs/wandb/",
        num_train_epochs=5,             # Total number of training epochs
        per_device_train_batch_size=16,  # Batch size per device during training
        per_device_eval_batch_size=32,   # Batch size for evaluation TODO: why it says this   File "/home/alisavin/AgenticADMET/train.py", line 534, in <module>
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
        gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
        learning_rate=1e-6,            # Initial learning rate for AdamW optimizer
        warmup_ratio=0.1,              # Linear warmup over warmup_ratio fraction of training steps
        weight_decay=0.01,             # Apply weight decay to all layers except bias and LayerNorm weights
        logging_steps=10,              # Log every X updates steps
        logging_strategy="steps",
        logging_first_step=True,
        evaluation_strategy="epoch",    # Evaluate every `eval_steps`
        save_strategy="epoch",      # Disables regular checkpoints
        save_total_limit=1,      # Makes sure no checkpoints are kept
        load_best_model_at_end=False,  # Disables saving the best model
        # save_steps=0,            # No saving at specific steps
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
        # eval_steps=10 #TODO: change to -1
        # - 1.0
        # - 1.0
        # push_to_hub=False,             # Whether to push the final model to Hugging Face Hub
    )
    
    #TODO: reward, for each property set threashold for MAE to set to range 0 to 1
    #TODO: loss always 0

    # Create GRPOConfig from TrainingArguments
    # grpo_config = GRPOConfig(
    #     **training_args.to_dict(), # Convert TrainingArguments to dictionary and unpack
    #     **{ 
    #     # REMOVED model_init_kwargs here 
    #     # We are passing the instantiated 'model' object, so GRPOTrainer doesn't need model_init_kwargs
    #     },
    #     num_generations=2, #TODO: 16
    #     use_vllm=False, #TODO: use True
    #     vllm_device="cuda:0",
    #     vllm_gpu_memory_utilization=0.3, # 0.7
    #     vllm_max_model_len=2048,
    #     # max_prompt_length=800, #TODO: 800+
    #     max_completion_length=1024, #TODO: 1024+ (better 2048/4048 and more)
    #     temperature=0.7,
    #     reward_weights=[1.0, 1.0, 0.4, 0.4]
    #     # packing=True
    #     )

    grpo_config = GRPOConfig(
        **training_args.to_dict(), # Convert TrainingArguments to dictionary and unpack
        **{ 
        # REMOVED model_init_kwargs here 
        # We are passing the instantiated 'model' object, so GRPOTrainer doesn't need model_init_kwargs
        },
        num_generations=16, #TODO: 16
        use_vllm=True, #TODO: use True
        vllm_device="cuda:0",
        vllm_gpu_memory_utilization=0.25, # TODO: 0.25 0.7
        vllm_max_model_len=2048, #TODO: 2048
        max_prompt_length=800, #TODO: 800+
        max_completion_length=1024, #TODO: 1024+ (better 2048/4048 and more)
        temperature=0.7,
        reward_weights=[1.0, 1.0, 0.4, 0.4]
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
        peft_config=get_peft_config(model_args) #TODO: check # label_names
    )
    print_trainable_parameters(grpo_trainer.model)

    # Start the GRPO Training Loop
    train_result = grpo_trainer.train()

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