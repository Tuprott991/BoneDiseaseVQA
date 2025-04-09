import csv
import sys

def clean_commas(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            cleaned_row = [cell.replace(',', '') for cell in row]
            writer.writerow(cleaned_row)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python question_cleaning.py input.csv output.csv")
    else:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]
        clean_commas(input_csv, output_csv)
        print(f"Cleaned CSV saved to {output_csv}")
