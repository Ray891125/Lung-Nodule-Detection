import csv

def writeCSV(filename, lines):
    with open(filename, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(lines)
def get_csv_titles(filename: str) -> list:
    """
    Reads a CSV file and returns the first row as a list of titles.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        list: A list of titles.
    """
    with open(filename, "r") as f:
        csv_reader = csv.reader(f)
        return next(csv_reader)
def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines
from itertools import islice

def read_csv_per_patient(filename: str, pid: str) -> list:
  """
  Reads a CSV file and returns lines that match a specific patient ID
  using itertools.islice for potentially better memory efficiency.

  Args:
      filename (str): The path to the CSV file.
      pid (str): The patient ID to match.

  Returns:
      list: A list (box float info ) of lines that match the patient ID.
  """
  with open(filename, "r") as f:
    csv_reader = csv.reader(f)
    return list(islice((row[1:] for row in csv_reader if row[0] == pid), None))

def tryFloat(value):
    try:
        value = float(value)
    except:
        value = value
    
    return value

def getColumn(lines, columnid, elementType=''):
    column = []
    for line in lines:
        try:
            value = line[columnid]
        except:
            continue
            
        if elementType == 'float':
            value = tryFloat(value)

        column.append(value)
    return column
