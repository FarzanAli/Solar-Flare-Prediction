import numpy as np

# Open the file and read its contents
# file_path = './data-2010-15/all_harps_with_noaa_ars.txt'  # replace with the actual file path
# harpnum_dict = {}

# with open(file_path, 'r') as file:
#     # Skip the first line (header)
#     next(file)
    
#     for line in file:
#         # Split the line by spaces and remove any extra spaces
#         parts = line.split()

#         if len(parts) == 2:  # Ensure there are two elements
#             harpnum = int(parts[0])
#             noaa_ars = parts[1].split(',')
            
#             # Add to the dictionary, with HARPNUM as the key and list as value
#             if harpnum not in harpnum_dict:
#                 harpnum_dict[harpnum] = []
            
#             harpnum_dict[harpnum] = noaa_ars

goes_data = np.load('./data-2010-15/goes_data.npy', allow_pickle=True)
res = filter(lambda x: x['noaa_active_region'] != 0, goes_data)
print(len(list(res)))

for event in goes_data:
    noaa_region = int(event['noaa_active_region'])


# Print the dictionary
# print(len(harpnum_dict))
