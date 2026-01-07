from chronos import BaseChronosPipeline, ChronosPipeline
import inspect
from visionFusion import BASE_MODEL_NAME

def main():
    print(f"Loading Base Pipeline: {BASE_MODEL_NAME}")
    try:
        pipeline = BaseChronosPipeline.from_pretrained(BASE_MODEL_NAME, device_map="cpu")
    except:
        # Fallback if BaseChronosPipeline cannot be instantiated directly or model name issues
        pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="cpu")
        
    print("\n--- Pipeline _predict_batch Source ---")
    try:
        # _predict_batch is likely a method of the pipeline instance
        if hasattr(pipeline, "_predict_batch"):
            print(inspect.getsource(pipeline._predict_batch))
        else:
            print("No _predict_batch method found.")
    except Exception as e:
        print(f"Could not get source: {e}")

if __name__ == "__main__":
    main()
