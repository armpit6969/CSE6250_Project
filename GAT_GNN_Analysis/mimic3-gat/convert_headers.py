import os
import csv

def convert_headers_to_lowercase(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and filename == "NOTEEVENTS.csv" or filename == "CHARTEVENTS.csv":
            file_path = os.path.join(directory, filename)
            temp_path = os.path.join(directory, f"temp_{filename}")
            
            with open(file_path, 'r', newline='', encoding='utf-8') as infile, \
                 open(temp_path, 'w', newline='', encoding='utf-8') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                
                headers = next(reader)
                lowercase_headers = [header.lower() for header in headers]
                writer.writerow(lowercase_headers)
                
                for row in reader:
                    writer.writerow(row)
            
            os.replace(temp_path, file_path)

if __name__ == "__main__":
    directory_path = 'physionet.org/files/mimiciii/1.4/'
    convert_headers_to_lowercase(directory_path)
