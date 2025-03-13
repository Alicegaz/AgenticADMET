from typing import Any

from chemprop.nn.metrics import MAE, R2Score, MSE
from chemprop.schedulers import build_NoamLike_LRSched
from lightning import pytorch as pl
import torch
from torch import nn
from torch import Tensor as T
from torch.nn import functional as F
from transformers import PretrainedConfig, RobertaModel

from utils import CheckpointParams, CheckpointDownloader


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim: int, num_layers: int, output_dim, 
                 bias_final: bool = True, dropout=0.0):
        super().__init__()
        fc_block = []
        for i in range(num_layers):
            input_dim = input_dim if i == 0 else hidden_dim
            
            layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                # nn.BatchNorm1d(hidden_dim)
            )
            fc_block.append(layer)
        
        self.fc_block = nn.Sequential(*fc_block)
        self.classifier = nn.Linear(hidden_dim, output_dim, bias=bias_final)

    def forward(self, x):
        x = self.fc_block(x)
        x = self.classifier(x)

        return x


class TransformerRegressionModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        model_params: dict[str, Any],
        weight_decay: float = 0.0,
        warmup_epochs: int = 10,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        checkpoints: list[CheckpointParams] | None = None,
    ) -> None:
        super().__init__()
        hparams = {
            'model_name': model_name,
            'model_params': model_params,
            'checkpoints': checkpoints,
            'weight_decay': weight_decay,
            'warmup_epochs': warmup_epochs,
            'init_lr': init_lr,
            'max_lr': max_lr,
            'final_lr': final_lr
        }
        self.save_hyperparameters(hparams)

        self._init_model(model_name, **model_params)
        self.criterion = MSE()
        self.metrics = nn.ModuleList([MAE(), R2Score()])

    def _init_model(self, model_name: str, **model_params):
        config = PretrainedConfig(**model_params['config'])
        model_params.pop('config')
        if model_name == 'roberta-base':
            self.roberta = RobertaModel(config, add_pooling_layer=False)
        else:
            raise ValueError(f'Model {model_name} is not supported')
        
        self.predictor = MLP(
            input_dim=self.roberta.config.hidden_size,
            hidden_dim=model_params['hidden_dim'],
            num_layers=model_params['num_layers'],
            output_dim=model_params['output_dim'],
            bias_final=model_params.get('bias_final', False),
            dropout=model_params.get('dropout', 0.0)
        )

        self.load_checkpoints()
        # self.roberta.requires_grad_(False)
    
    def forward(self, input_ids: T, attention_mask: T) -> T:
        outputs = self.roberta.forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        embeddings = outputs['last_hidden_state'][:, 1:, :]
        attention_mask = attention_mask[:, 1:]

        # embeddings = (embeddings * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).type(torch.bool)
        embeddings[~input_mask_expanded] = -1e9  # Set padding tokens to large negative value
        embeddings = F.max_pool1d(embeddings.permute(0, 2, 1), kernel_size=embeddings.size(1)).squeeze(-1)

        preds = self.predictor(embeddings)

        return preds

    def training_step(self, batch: Any, batch_idx: int) -> T:
        mol_input_ids, attention_mask, targets, weights = \
            batch['input_ids'], batch['attention_mask'], batch['targets'], batch['weights']

        preds = self.forward(mol_input_ids, attention_mask)
        loss = self._calc_loss(preds, targets, weights)

        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> T:
        mol_input_ids, attention_mask, targets, weights = \
            batch['input_ids'], batch['attention_mask'], batch['targets'], batch['weights']
        
        preds = self.forward(mol_input_ids, attention_mask)
        loss = self._calc_loss(preds, targets, weights)

        self._evaluate_batch('val', preds, targets)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> T:
        mol_input_ids, attention_mask, targets, weights = \
            batch['input_ids'], batch['attention_mask'], batch['targets'], batch['weights']
        
        preds = self.forward(mol_input_ids, attention_mask)

        self._evaluate_batch('test', preds, targets)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> T:
        mol_input_ids, attention_mask = \
            batch['input_ids'], batch['attention_mask']
        
        preds = self.forward(mol_input_ids, attention_mask)

        return preds
    
    def configure_optimizers(self):
        # Group parameters by decay and no_decay. Implementation is from Hugging Face Transformers: 
        # https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/trainer.py#L923
        decay_parameters = TransformerRegressionModel._get_parameter_names(self, forbidden_layer_types=[nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if n in decay_parameters],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters,
            eps=1e-6,
            lr=self.hparams.init_lr,
            weight_decay=self.hparams.weight_decay
        )

        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.hparams.warmup_epochs * steps_per_epoch
        cooldown_epochs = self.trainer.max_epochs - self.hparams.warmup_epochs
        cooldown_steps = cooldown_epochs * steps_per_epoch

        scheduler = build_NoamLike_LRSched(
            optimizer,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            init_lr=self.hparams.init_lr,
            max_lr=self.hparams.max_lr,
            final_lr=self.hparams.final_lr
        )

        return [optimizer], [scheduler]
        # return optimizer
    
    @staticmethod
    def _get_parameter_names(model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        (Implementation is from Hugging Face Transformers: 
        https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/trainer_pt_utils.py)
        """
        result = []
        for name, child in model.named_children():
            result += [
                f'{name}.{n}'
                for n in TransformerRegressionModel._get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        
        return result
    
    def _calc_loss(self, preds: T, targets: T, weights: T) -> T:
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        loss = self.criterion(preds=preds, targets=targets, mask=mask, weights=weights)

        return loss

    def _evaluate_batch(self, stage: str, preds: T, targets: T) -> None:
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        weights = torch.ones(targets.shape[0], dtype=torch.float32, device=targets.device)

        for m in self.metrics:
            m.update(preds=preds, targets=targets, mask=mask, weights=weights)
            self.log(f"{stage}/{m.alias}", m)
    
    def load_checkpoints(self) -> None:
        """Load model weights from a list of checkpoints to submodules of self."""
        checkpoints_params = self.hparams.get('checkpoints', None)
        if checkpoints_params is None:
            print(f'No checkpoint is provided for {self.__class__.__name__}')
            return

        # if full_checkpoint is provided, then override given checkpoints and load the full checkpoint
        # (this is useful for cases when resuming training while changing some parameters like learning rate 
        # as this isn't supported by PyTorch Lightning)
        full_checkpoint_params = self.hparams.get('full_checkpoint', None)
        if full_checkpoint_params is not None:
            print(f'Overriding given checkpoints with full checkpoint from {full_checkpoint_params.path}...')
            checkpoints_params = [CheckpointParams(
                path=full_checkpoint_params.path,
                strict=full_checkpoint_params.strict,
            )]

        for checkpoint_params in checkpoints_params:
            self._load_checkpoint_for_module(checkpoint_params)

    def _load_checkpoint_for_module(self, checkpoint_params: CheckpointParams) -> None:
        """Load model weights from checkpoint."""
        print(f'Getting checkpoint from {checkpoint_params.path}...')

        with CheckpointDownloader(checkpoint_params.path) as downloader:
            checkpoint = torch.load(downloader.path, weights_only=False, map_location='cpu')

        # Get prefix which should be removed from the checkpoint keys
        if checkpoint_params.module_from is not None:
            prefix_from = checkpoint_params.module_from + '.' \
                if not checkpoint_params.module_from.endswith('.') \
                else checkpoint_params.module_from
        else:
            prefix_from = ''

        # Loading full checkpoint for a specific module
        if checkpoint_params.module_to is not None:
            module_is_found = False
            for module_name, module in self.named_modules():
                if module_name == checkpoint_params.module_to:
                    print(f'Loading checkpoint from {checkpoint_params.module_from or "full checkpoint"} ' \
                          f'to {module_name}...')
                    state_dict = {
                        k.replace(prefix_from, ''): v
                        for k, v in checkpoint['state_dict'].items()
                        if k.startswith(prefix_from)
                    }   # remove prefix
                    print(module.load_state_dict(state_dict, strict=checkpoint_params.strict))
                    module_is_found = True
                    break

            if not module_is_found:
                raise ValueError(f'Module {checkpoint_params.module_to} is not found in the model')
        else:
            print(f'Loading checkpoint for the whole model...')
            print(self.load_state_dict(checkpoint['state_dict'], strict=checkpoint_params.strict))
