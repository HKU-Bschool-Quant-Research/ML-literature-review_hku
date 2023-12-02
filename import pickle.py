import pickle

# 读取.pkl文件
with open('AgesAndHeights.pkl', 'rb') as file:
    data = pickle.load(file)

# 使用导入的数据
print(data)