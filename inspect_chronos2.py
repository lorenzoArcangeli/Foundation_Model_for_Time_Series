
import sys
import torch
try:
    from chronos.chronos2.pipeline import Chronos2Pipeline
    print("Successfully imported Chronos2Pipeline")
    
    print("\n=== Chronos2Pipeline.predict signature ===")
    print(Chronos2Pipeline.predict.__doc__)
    
    # Try to load a config or small model to see the model class structure
    # If loading is too heavy, we might just inspect the class itself for hints
    # but inspecting the model's forward is best.
    # We'll try to load the smallest one if possible, or just skip if it requires downloading big weights.
    # But usually we need the model object to see forward().
    
    print("\n=== Inspecting Model Forward ===")
    # It seems we can't easily instantiate without downloading. 
    # Let's check if we can inspect the class of the model property if it's annotated.
    
    # Alternative: check if there's a reference to the model class in the pipeline class
    
except ImportError as e:
    print(f"Failed to import Chronos2Pipeline: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
