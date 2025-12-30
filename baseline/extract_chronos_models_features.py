import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from chronos import ChronosPipeline, BaseChronosPipeline
# Try importing Chronos2Pipeline if available, to verify class type
try:
    from chronos.chronos2.pipeline import Chronos2Pipeline
except ImportError:
    Chronos2Pipeline = None


def get_pipeline(model_name):
  if "chronos-2" in model_name:
    return Chronos2Pipeline
  elif "bolt" in model_name:
    return BaseChronosPipeline
  return ChronosPipeline

def get_correct_model_input_size(model_name, context_list):
  if "chronos-2" in model_name:
      return context_list.unsqueeze(0).unsqueeze(0) # (1, 1, 512)
  return context_list.unsqueeze(0) # (1, 512)

def get_prediction(model_name, pipeline, model_input):
  return pipeline.predict(
                model_input,
                prediction_length=PREDICTION_LENGTH,
            )
  
def extract_embeddings_chronos2(pipeline, context_tensor):
    """
    Manual embedding extraction for Chronos-2 which lacks pipeline.embed()
    """
    model = pipeline.model
    device = model.device
    context_tensor = context_tensor.to(device)
    
    # 1. Prepare Patched Context
    # We need to call internal methods. 
    # _prepare_patched_context(context, context_mask=None)
    # It returns patched_context, attention_mask, loc_scale
    
    # Check signature of _prepare_patched_context via inspection or assumption from code reading
    # method signature: (self, context, context_mask, batch_size)
    # Wait, looking at code snippet: def _prepare_patched_context(self, context, context_mask, batch_size):
    
    batch_size = context_tensor.shape[0]
    
    with torch.no_grad():
        patched_context, attention_mask, loc_scale = model._prepare_patched_context(
            context=context_tensor,
            context_mask=None,
            #batch_size=batch_size
        )
        
        # 2. Input Embeddings
        # input_embeds = model.input_patch_embedding(patched_context)
        input_embeds = model.input_patch_embedding(patched_context)

        # Mask should be numeric.
        attention_mask = attention_mask.to(dtype=torch.long)
        
        # 3. Encoder
        # encoder_outputs = model.encoder(inputs_embeds=input_embeds, attention_mask=attention_mask, ...)
        
        # We need to construct group_ids if needed (default None is handled in forward, but encoder might need it?)
        # chunk 6 says: if group_ids is None: group_ids = torch.arange(...)
        group_ids = torch.arange(batch_size, dtype=torch.long, device=device)
        
        encoder_outputs = model.encoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            group_ids=group_ids
        )
        
        # encoder_outputs is likely a specialized object or HF standard. 
        # Usually has .last_hidden_state
        return encoder_outputs.last_hidden_state



# Configuration
DATA_PATH = "/content/drive/MyDrive/FM_project/dataset/skippd_train_no_images.parquet"
MODELS_TO_RUN = ["amazon/chronos-t5-small", "amazon/chronos-bolt-small","amazon/chronos-bolt-base", "amazon/chronos-2"]
#MODELS_TO_RUN=["amazon/chronos-2"]
CONTEXT_LENGTH = 600 
PREDICTION_LENGTH = 60

def main():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_parquet(DATA_PATH)
        
        # Ensure 'pv' column exists
        if 'pv' not in df.columns:
            raise ValueError("Column 'pv' not found in dataset")
        
        full_series = df["pv"].values

        # Dataset Split
        total_len = len(full_series)
        split_point = total_len - PREDICTION_LENGTH
        context_start = split_point - CONTEXT_LENGTH
        
        # Extract tensors
        context_tensor = full_series[context_start : split_point]
        ground_truth = full_series[split_point : total_len]
        
        context_list = torch.tensor(context_tensor, dtype=torch.float32)#[context_tensor]
            
        print(f"Context range: {context_start} to {split_point}")
        print(f"Prediction range: {split_point} to {total_len}")

        # Prepare X-axis (Time)
        if 'time' in df.columns:
            time_series = pd.to_datetime(df['time'].values)
            x_context = time_series[context_start:split_point]
            x_future = time_series[split_point:total_len]
        else:
            x_context = np.arange(context_start, split_point)
            x_future = np.arange(split_point, total_len)

        # Plot Setup
        fig, axes = plt.subplots(nrows=len(MODELS_TO_RUN), ncols=1, figsize=(12, 6 * len(MODELS_TO_RUN)), sharex=False)
        if len(MODELS_TO_RUN) == 1: axes = [axes]
        
        for i, model_name in enumerate(MODELS_TO_RUN):
            print(f"\n--- Processing Model: {model_name} ---")
            ax = axes[i]
            
            # Select correct pipeline class
            pipeline_class=get_pipeline(model_name)

            pipeline = pipeline_class.from_pretrained(
                model_name,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
            )

            model_input=get_correct_model_input_size(model_name, context_list)
            
            # 1. Forecasting
            forecast = get_prediction(model_name, pipeline, model_input)

            
            # 2. Embedding Extraction
            #embeddings, _ = pipeline.embed(model_input)
            if "chronos-2" in model_name or "710m-2" in model_name: # Handle Chronos 2 specifically
                 #embeddings = extract_embeddings_chronos2(pipeline, model_input)
                 embeddings = extract_embeddings_chronos2(pipeline, context_list.unsqueeze(0))
            else:
                 # Standard T5/Bolt
                 # Bolt expects tensor, T5 expects list/tensor
                 embeddings, _ = pipeline.embed(model_input)

            print(f"Embeddings shape: {embeddings.shape}")
            
            # Save Embeddings
            safe_name = model_name.replace("/", "_").replace("-", "_")
            save_path = f"embeddings_{safe_name}.pt"
            torch.save(embeddings, save_path)
            print(f"Saved embeddings to '{save_path}'")
            
            # 3. Visualization Logic
            forecast_numpy = forecast[0].numpy() 
            
            # Shape Fixer
            if forecast_numpy.ndim == 3 and forecast_numpy.shape[0] == 1:
                forecast_numpy = forecast_numpy[0]
            if forecast_numpy.ndim == 3 and forecast_numpy.shape[-1] == 1:
                forecast_numpy = forecast_numpy[..., 0]

            # Stats
            median_forecast = np.median(forecast_numpy, axis=0)
            low_quant = np.quantile(forecast_numpy, 0.1, axis=0)
            high_quant = np.quantile(forecast_numpy, 0.9, axis=0)
            
            # Plot
            plot_len = 100
            ax.plot(x_context[-plot_len:], context_tensor[-plot_len:], label="Context", color="black", alpha=0.5)
            ax.plot(x_future, ground_truth, label="Ground Truth", color="green", linewidth=2)
            ax.plot(x_future, median_forecast, label="Median Forecast", color="blue", linestyle="--")
            ax.fill_between(x_future, low_quant, high_quant, color="blue", alpha=0.2, label="80% CI")
            
            ax.set_title(f"Model: {model_name}")
            
            if 'time' in df.columns:
                date_format = mdates.DateFormatter('%d-%m %H:%M')
                ax.xaxis.set_major_formatter(date_format)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_plot = 'chronos_comparison.png'
        plt.savefig(output_plot)
        print(f"\nSaved comparison plot to '{output_plot}'")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
