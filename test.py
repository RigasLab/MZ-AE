# import torch

# x = torch.arange(1,37).reshape(3,3,4)
# x = torch.movedim(x,-1,1)
# y = torch.flatten(x, start_dim = 0, end_dim = 1)
# c = torch.arange(1,13).reshape(4,3)
# z = y.reshape(x.shape)
# # z = y.reshape(3,3,3)
# print(x)
# print(y.shape)
# print(z)
# print(z.shape)
# print(c.shape)
# print((3,*c.shape))

import csv

data = [
    {'column1': 'value1', 'column2': 'value2', 'column3': 'value3'},
    {'column1': 'value4', 'column2': 'value5', 'column3': 'value6'}
]

filename = 'output.csv'
fieldnames = ['column1', 'column2', 'column3']

# Open the CSV file for writing
with open(filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Format and write the header with spaces between column names
    formatted_header = {key: f'{key: <10}' for key in fieldnames}
    writer.writerow(formatted_header)
    
    # Write the data with spaces between columns
    for row in data:
        formatted_row = {key: f'{value: <10}' for key, value in row.items()}
        writer.writerow(formatted_row)

print(f'Data has been written to {filename}')
