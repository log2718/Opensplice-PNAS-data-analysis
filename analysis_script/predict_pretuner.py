'''
Reads test_exons.csv, computes PNAS inputs for the exon variants, and extracts the pretuner prediction.
Stores data into a CSV file.
'''

from pathlib import Path
import torch
import sys
import pandas as pd  

#------
# Paths
#------
ROOT = Path(__file__).resolve().parent.parent
PNAS_DIR = ROOT / "PNAS_model"
sys.path.insert(0, str(ROOT))
from PNAS_model.model import PNASModel 
from PNAS_model.utils import create_input_data 

INPUT_CSV = ROOT / "outputs" / "test_exons.csv"
OUTPUT_CSV = ROOT / "outputs" / "test_exons_with_pretuner.csv"
WEIGHTS = PNAS_DIR / "model_weights.pt"


#------
# Pretuner Forward Pass
#------
@torch.no_grad()
def get_pretuner(model, x_seq, x_struct, x_wobble):
    """Compute exon inclusion probabilities, return the pretuner value. Similar to PNAS forward().

    Args:
        model: PNASModel
        x_seq: Sequence tensor of shape ``(batch_size, 4, input_length)``.
        x_struct: Structure tensor of shape ``(batch_size, 3, input_length)``.
        x_wobble: Wobble tensor of shape ``(batch_size, 1, input_length)``.
    """
    # Compute sequence activations - each is (batch_size, F_seq, 85)
    conv_skip_out = model.conv_skip(x_seq) + model.position_bias_skip.unsqueeze(0)  # Add position bias
    conv_incl_out = model.conv_incl(x_seq) + model.position_bias_incl.unsqueeze(0)
    
    # Compute structure activations - each is (batch_size, F_struct, 90)
    struct_input = torch.cat([x_seq, x_struct, x_wobble], dim=1)  # Concatenate along channel dimension
    conv_struct_skip_out = model.conv_struct_skip(struct_input) + model.position_bias_skip_struct.unsqueeze(0)
    conv_struct_incl_out = model.conv_struct_incl(struct_input) + model.position_bias_incl_struct.unsqueeze(0)

    # Crop to match sequence activations
    conv_struct_skip_out = conv_struct_skip_out[:, :, 2:-3]
    conv_struct_incl_out = conv_struct_incl_out[:, :, 2:-3]

    # Concatenated activations
    activations_skip = model.energy_activation_skip(torch.cat([conv_skip_out, conv_struct_skip_out], dim=1))  # (batch_size, F_seq + F_struct, L-5)
    activations_incl = model.energy_activation_incl(torch.cat([conv_incl_out, conv_struct_incl_out], dim=1))  # (batch_size, F_seq + F_struct, L-5)

    # Apply sum-difference
    energy_in = torch.stack([activations_incl, activations_skip], dim=1)  # (batch_size, 2, F_seq + F_struct, L-5)
    energy_out = model.energy_seq_struct(energy_in)

    # Apply tuner
    #tuner_out = self.tuner(energy_out)  # (batch_size, 1)

    #if return_logits:
    #    return tuner_out.squeeze(-1)  # (batch_size,)
    
    # compute sigmoid, return (0, 1)
    #out = self.output_activation(tuner_out).squeeze(-1)  # (batch_size,)

    return energy_out

#------
#Main
#The input will be the exon without the flanks used in the Opensplice experiments
#------
def main():
    df = pd.read_csv(INPUT_CSV) 
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    all_results = [] 

    #Different exons have different lengths, needs separate processing
    df["model_input_length"] = df["exon"].astype(str).str.len()

    for input_length, group in df.groupby("model_input_length"):
        print(
            f"Processing exon length {input_length}: "
            f"{len(group)} variants"
        )

        #get mutant exons
        sequences = group["exon"].astype('str').tolist() 

        seq_oh, struct_oh, wobbles = create_input_data(
            sequences,
            add_flanks=False,
            temperature=37.0,
            num_threads=8,
        )

        x_seq = torch.tensor(
            seq_oh,
            dtype=torch.float32,
            device=device,
        )
        x_struct = torch.tensor(
            struct_oh,
            dtype=torch.float32,
            device=device,
        )
        x_wobble = torch.tensor(
            wobbles,
            dtype=torch.float32,
            device=device,
        )
        #input length = exon length
        input_length = x_seq.shape[-1]
        model = PNASModel(input_length=input_length)
        checkpoint = torch.load(
            WEIGHTS,
            map_location="cpu",
            weights_only=False,
        )
        
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint,
        )

        # This will automatically resample the learned position biases to the current exon length
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval() 

        pretuner = get_pretuner(
            model,
            x_seq,
            x_struct,
            x_wobble,
        )

        #results
        result = group.copy()
        result['pretuner'] = (
            pretuner.detach().cpu().numpy()
        )
        all_results.append(result)

    #put exons of different lengths back together 
    result_df = pd.concat(
        all_results,
        ignore_index=True,
    )
    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(f"\nSaved predictions to:\n{OUTPUT_CSV}")
    print(
        result_df[
            [
                "exon_id",
                "variant_id",
                "wt_exon_length",
                "psi",
                "logit_psi",
                "pretuner",
            ]
        ].head(10)
    )

if __name__ == "__main__":
    main()




