import json

def clean_data(input_file, output_file):
    # Load the JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter out entries with diagnose == "NotFound"
    cleaned_data = [entry for entry in data if entry.get("diagnose") != "NotFound"]

    # Save the cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

    print(f"Cleaned data saved to '{output_file}'. Removed {len(data) - len(cleaned_data)} entries.")

if __name__ == "__main__":
    input_path = 'dataset/output_bonedata.json'          # Replace with your actual input file name
    output_path = 'dataset/cleaned_output_bonedata.json' # Output file name
    clean_data(input_path, output_path)
