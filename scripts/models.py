import torch.nn as nn


class MoLFormerWithRegressionHeadMLM(nn.Module):
    def __init__(self, language_model):
        super().__init__()

        self.language_model = language_model
        
        for param in self.language_model.parameters():
            param.requires_grad = True

        self.layer = nn.Sequential(
            # Layer 1
            nn.Linear(768, 407),
            nn.BatchNorm1d(407),
            nn.ELU(),
            nn.Dropout(0.3819889369374411),
            # Layer 2
            nn.Linear(407,427),
            nn.BatchNorm1d(427),
            nn.ELU(),
            nn.Dropout(0.2777819584361112),
            # Layer 3
            nn.Linear(427,240),
            nn.BatchNorm1d(240),
            nn.ELU(),
            nn.Dropout(0.4619253799146514),
            # Layer 4
            nn.Linear(240, 69),
            nn.BatchNorm1d(69),
            nn.ELU(),
            nn.Dropout(0.4497051723910867),
            nn.Linear(69,1)
        )
    
    def forward(self, token):

        smiles_output = self.language_model(**token, output_hidden_states=True)
        smiles_embedding = smiles_output.hidden_states[-1][:, 0, :].float()
        
        return self.layer(smiles_embedding)