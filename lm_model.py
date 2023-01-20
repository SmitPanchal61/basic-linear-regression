import pandas
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model

input_data = pandas.read_table("height.csv", sep=",", header=0, names=("weight","height"))

plt.scatter(input_data["weight"], input_data["height"])
# plt.show()

predictor = pd.DataFrame(input_data, columns=["weight"])
outcome = pd.DataFrame(input_data, columns=["height"])

lm = linear_model.LinearRegression()
lm_model = lm.fit(predictor, outcome)

predicted_heights = lm.predict(predictor)

# print("predicted:")
# print(predicted_heights[0:6])
# print("actual:")
# print(outcome[0:6])

r_squared = lm.score(predictor, outcome)
print(r_squared)