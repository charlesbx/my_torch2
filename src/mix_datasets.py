import numpy as np
from TrainingSet import TrainingSet, TrainingSet64Inputs

def retrieve_data_from_dataset(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        i = 0
        start = i
        content = []
        while i < len(lines):
            if lines[i] != "\n":
                i += 1
                continue
            else:
                content.append(lines[start:i])
                start = i + 1
                i += 1
    return content

# checkmate concat   
# content1 = retrieve_data_from_dataset('datasets/checkmate/10_pieces.txt')
# content2 = retrieve_data_from_dataset('datasets/checkmate/20_pieces.txt')
# content3 = retrieve_data_from_dataset('datasets/checkmate/lots_pieces.txt')

# full_content = content1 + content2 + content3
# np.random.shuffle(full_content)

# with open('datasets/checkmate/all_checkmate.txt', 'w') as f:
#     for content in full_content:
#         for line in content:
#             f.write(line)
#         f.write("\n")

# boards concat   
# content1 = retrieve_data_from_dataset('datasets/boards/10_pieces.txt')
# content2 = retrieve_data_from_dataset('datasets/boards/20_pieces.txt')
# content3 = retrieve_data_from_dataset('datasets/boards/lots_pieces.txt')

# full_content = content1 + content2 + content3
# np.random.shuffle(full_content)

# with open('datasets/boards/all_boards.txt', 'w') as f:
#     for content in full_content:
#         for line in content:
#             f.write(line)
#         f.write("\n")
        
# all_dataset concat
# content1 = retrieve_data_from_dataset('datasets/checkmate/all_checkmate.txt')
# content2 = retrieve_data_from_dataset('datasets/boards/all_boards.txt')

# full_content = content1 + content2
# np.random.shuffle(full_content)

# with open('datasets/both/all_dataset.txt', 'w') as f:
#     for content in full_content:
#         for line in content:
#             f.write(line)
#         f.write("\n")

# 10_pieces concat
# content1 = retrieve_data_from_dataset('datasets/boards/10_pieces.txt')
# content2 = retrieve_data_from_dataset('datasets/checkmate/10_pieces.txt')

# full_content = content1 + content2
# np.random.shuffle(full_content)

# with open('datasets/both/10_pieces.txt', 'w') as f:
#     for content in full_content:
#         for line in content:
#             f.write(line)
#         f.write("\n")
        
# 20_pieces concat
# content1 = retrieve_data_from_dataset('datasets/boards/20_pieces.txt')
# content2 = retrieve_data_from_dataset('datasets/checkmate/20_pieces.txt')

# full_content = content1 + content2
# np.random.shuffle(full_content)

# with open('datasets/both/20_pieces.txt', 'w') as f:
#     for content in full_content:
#         for line in content:
#             f.write(line)
#         f.write("\n")

# lots_pieces concat
# content1 = retrieve_data_from_dataset('datasets/boards/lots_pieces.txt')
# content2 = retrieve_data_from_dataset('datasets/checkmate/lots_pieces.txt')

# full_content = content1 + content2
# np.random.shuffle(full_content)

# with open('datasets/both/lots_pieces.txt', 'w') as f:
#     for content in full_content:
#         for line in content:
#             f.write(line)
#         f.write("\n")

# print shapes
# Both 10 pieces        (196909, 384)
# Both 20 pieces        (557431, 384)
# Both lots pieces      (245669, 384)
# Both all pieces       (1000009, 384)
# Boards all pieces     (963704, 384)
# Checkmate all pieces  (36305, 384)
# X, y = TrainingSet('datasets/both/10_pieces.txt').get_formatted_data()
# print("Both 10 pieces")
# print(X.shape)
# print(y.shape)
# X, y = TrainingSet('datasets/both/20_pieces.txt').get_formatted_data()
# print("Both 20 pieces")
# print(X.shape)
# print(y.shape)
# X, y = TrainingSet('datasets/both/lots_pieces.txt').get_formatted_data()
# print("Both lots pieces")
# print(X.shape)
# print(y.shape)
# X, y = TrainingSet('datasets/both/all_dataset.txt').get_formatted_data()
# print("Both all pieces")
# print(X.shape)
# print(y.shape)
# X, y = TrainingSet('datasets/boards/all_boards.txt').get_formatted_data()
# print("Boards all pieces")
# print(X.shape)
# print(y.shape)
# X, y = TrainingSet('datasets/checkmate/all_checkmate.txt').get_formatted_data()
# print("Checkmate all pieces")
# print(X.shape)
# print(y.shape)

# create reference dataset
# content1 = retrieve_data_from_dataset('datasets/boards/all_boards.txt')
# np.random.shuffle(content1)
# content1 = content1[:100000]
# with open('datasets/both/reference_dataset_100000.txt', 'w') as f:
#     for content in content1:
#         for line in content:
#             f.write(line)
#         f.write("\n")