import torch
import torch.nn as nn

from models import MoLFormerWithRegressionHeadMLM
from transformers import AutoModel, AutoTokenizer


class IA3MolFormer(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.modify_mlp_layers()
        self.modify_attention_layers()

    def modify_mlp_layers(self):
        for module in self.model.layer:
            if isinstance(module, nn.Linear):

                module.ia3_alpha = nn.Parameter(torch.ones(module.out_features, requires_grad=True))

                original_forward = module.forward
                def ia3_forward(x, alpha = module.ia3_alpha):
                    return original_forward(x * alpha)
                
                module.forward = ia3_forward

    def modify_attention_layers(self):

        for layer in self.model.language_model.encoder.layer:
            attn = layer.attention.self

            attn.ia3_alpha_k = nn.Parameter(torch.ones_like(attn.key.weight, requires_grad=True))  
            attn.ia3_alpha_v = nn.Parameter(torch.ones_like(attn.value.weight, requires_grad=True))

            def modify_kv_projection(original_forward, alpha):
                return lambda x: original_forward(x) * alpha
            

            attn.key.forward = modify_kv_projection(attn.key.forward, attn.ia3_alpha_k)
            attn.value.forward = modify_kv_projection(attn.value.forward, attn.ia3_alpha_v)

    def forward(self, token):
        return self.model(token)




MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
raw_language_model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)

model = MoLFormerWithRegressionHeadMLM(raw_language_model)

ia3_model = IA3MolFormer(model)

for name, param in ia3_model.named_parameters():
    print(f"{name}: requires_grad = {param.requires_grad}")
