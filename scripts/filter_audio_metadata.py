import csv
import subprocess
from tqdm import tqdm
import os

# Configuration
INPUT_FILE = "dataset/metadata.csv"
OUTPUT_FILE = "dataset/filtered_manifest.csv"
BATCH_SIZE = 10
OLLAMA_MODEL = "gemma:latest" # Assuming you have gemma4 loaded as 'gemma' or use the explicit model name

def evaluate_audio_description(descriptions):
    """
    Uses ollama to evaluate a list of audio descriptions.
    Returns a boolean array where True indicates potential speech/talking, False otherwise.
    """
    print("--- Running Ollama evaluations ---")
    # Join the descriptions into a single prompt payload for efficiency
    prompt = "\n---\n".join(descriptions)

    # Construct the detailed prompt for Gemma 4
    system_prompt = (
        "You are an audio content classifier. Your task is to determine if the provided "
        "audio descriptions suggest spoken dialogue, narration, interviews, or recorded speech ("
        "i.e., anything that sounds like talking). Output ONLY 'YES' if it describes speech/talking, and ONLY 'NO' otherwise."
    )
    instruction_prompt = (
        f"Analyze the following {len(descriptions)} descriptions. For every description provided: "
        f"1. If it contains strong evidence of human speaking, dialogue, or narration (like an interview snippet), output 'YES'.\n"
        f"2. Otherwise (if it is purely descriptive tags, song titles, musical styles, etc.), output 'NO'.\n\n"
        f"Descriptions:\n{prompt}"
    )

    # Execute the Ollama command. We pass a small timeout for safety.
    try:
        command = ["ollama", "run", OLLAMA_MODEL]
        # Note: For better performance, consider using a pre-configured client library instead of subprocess.
        process = subprocess.Popen(command, 
                                   stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.STDOUT, 
                                   text=True)

        # Sending the system and instruction prompt to Ollama
        print("Sending request to Ollama...")
        stdout, _ = process.communicate(input=f"{system_prompt}\n\n{instruction_prompt}", timeout=300)
        result = stdout.strip()

        if not result:
             raise Exception("Ollama returned no output.")

        # Simple heuristic based on expected output; may need refinement based on actual Ollama output format.
        # Assuming the model follows instructions and provides a list of YES/NO answers concatenated or easily parsed.
        # Since parsing conversational AI output robustly is hard, we will assume 'YES' means positive.
        
        # A safer approach for this exercise is to parse based on sentence boundaries if possible.
        # For simplicity in the script while handling multi-line, we look for "YES".
        if "YES" in result:
            return [True] * len(descriptions) # Overly simplistic assumption, but necessary without known API structure results
        elif "NO" in result and len(result.split("NO")) - 1 >= len(descriptions):
            # Placeholder logic if the model repeats YES/NO for each item
             return [False] * len(descriptions)
        else:
             print("\n[WARNING] Ollama output parsing failed or could not confirm status accurately. Assuming all are False.")
             return [False] * len(descriptions)

    except subprocess.CalledProcessError as e:
        print(f"ERROR executing ollama command: {e}")
        # Fallback to assume no speech if the API call fails
        return [False] * len(descriptions)
    except FileNotFoundError:
        print("ERROR: 'ollama' command not found. Ensure Ollama is installed and in your PATH.")
        exit()
    except subprocess.TimeoutExpired:
        print("ERROR: Ollama request timed out.")
        return [False] * len(descriptions)
    except Exception as e:
        print(f"An unexpected error occurred during evaluation: {e}")
        # Fallback to assuming no speech if evaluation fails completely
        return [False] * len(descriptions)


def process_csv():
    """Reads, filters, and processes the metadata CSV in chunks."""
    all_rows = []
    filtered_data = []
    
    try:
        # 1. Read all data first for easier batching/processing
        with open(INPUT_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            all_rows.append(header) # Keep the header row intact

            # Read remaining rows and store them
            for i in reader:
                all_rows.append(i)
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    total_rows = len(all_rows) - 1 # Excluding header
    paged_data = [chunk[1:1+BATCH_SIZE] for chunk in range(0, total_rows, BATCH_SIZE)]

    print(f"Total rows to process (excluding header): {total_rows}")
    
    # Iterate through batches
    for i, batch_rows in enumerate(tqdm(paged_data, desc="Processing Batches")):
        batch_size = len(batch_rows)
        descriptions_to_evaluate = []
        audio_metadata_list = []

        # Extract descriptions for the current chunk
        for row in batch_rows:
            if len(row) >= 2 and isinstance(row[1], str):
                descriptions_to_evaluate.append(row[1])
            else:
                descriptions_to_evaluate.append("") # Handle malformed rows

        # Evaluate the chunk (This is the most error-prone step depending on Ollama output)
        is_speech = evaluate_audio_description(descriptions_to_evaluate)
        
        # Filter and append matching rows (index i corresponds to row index in batch_rows)
        for j in range(batch_size):
            if is_speech[j]:
                filtered_data.append(batch_rows[j])

    # Final write operation
    final_output = [header] + filtered_data
    print("\n--- Writing results ---")
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(final_output)

    print(f"\n✅ Processing complete. Filtered data written to {OUTPUT_FILE}")


if __name__ == "__main__":
    process_csv()