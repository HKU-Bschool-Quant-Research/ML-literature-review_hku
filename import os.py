import pickle
# 读取.pkl文件
with open('/Users/john/Documents/QuantML /FinRL/examples/A.pkl', 'rb') as file:
    data = pickle.load(file)

# 使用导入的数据
print(data)
import pandas as pd
raw_data = pd.read_pickle('/Users/john/Documents/QuantML /FinRL/examples/A.pkl')
import pandas as pd
import matplotlib.pyplot as plt

description = raw_data.describe()
print(description)
cleaned_data = raw_data[raw_data['Age'] > 0]
print(cleaned_data)
correlation_matrix = cleaned_data.corr()
print(correlation_matrix)
ages = cleaned_data['Age']
heights = cleaned_data['Height']

plt.scatter(ages, heights, label='Raw Data')
plt.title('Height VS Age')
plt.xlabel('Age [Years]')
plt.ylabel('Height [Inches]')
plt.legend()
plt.show()
parameters = {'alpha' : 40 ,'beta' : 4}
def y_hat(age, params):
  alpha = params['alpha']
  beta = params['beta']
  return alpha + beta * age
age = int(input('Enter age: '))
y_hat(age, parameters)
def learn_parameters(data, params):
    x, y = data['Age'], data['Height']
    x_bar, y_bar = x.mean(), y.mean()
    x, y = x.to_numpy(), y.to_numpy()
    beta = sum( ((x-x_bar) * (y-y_bar)) / sum( (x-x_bar)**2))
    alpha = y_bar - beta * x_bar
    params['alpha'] = alpha
    params['beta'] = beta
new_parameter = {'alpha' : -2, 'beta' : 1000}
learn_parameters(cleaned_data, new_parameter)
print(new_parameter)
spaced_ages = list(range(19))
spaced_untrained_predictions = [y_hat(x, parameters) for x in spaced_ages]
print(spaced_untrained_predictions)
ages = cleaned_data['Age']
heights = cleaned_data['Height']
ages = cleaned_data['Age']
heights = cleaned_data['Height']
plt.scatter(ages,heights, label='Raw Data')
plt.plot(spaced_ages, spaced_untrained_predictions, label = 'Untrained Predictions', color = 'green')
plt.title('Height VS Age')
plt.xlabel('Age[Years]')
plt.ylabel('Height[Inches]')
plt.legend()
plt.show()
spaced_trained_predictions = [y_hat(x, new_parameter) for x in spaced_ages]
print('Trained Predicted Values: ',spaced_trained_predictions)
plt.scatter(ages,heights, label='Raw Data')
plt.plot(spaced_ages, spaced_untrained_predictions, label = 'Untrained Predictions', color = 'green')
plt.plot(spaced_ages, spaced_trained_predictions, label = 'Trained Predictions', color = 'red')
plt.title('Height VS Age')
plt.xlabel('Age[Years]')
plt.ylabel('Height[Inches]')
plt.legend()
plt.show()
new_age = int(input('Enter age to predict height: '))
print(y_hat(new_age, new_parameter))