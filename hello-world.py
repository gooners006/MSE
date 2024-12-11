# Question 15: Assume we have a string "Today is Sunday and we don't need to wake up at 6 am".
# Print how many words in the string and check whether number in string. Print position of that number in string

# string = "Today is Sunday and we don't need to wake up at 6 am 6"
# WordList = string.split(" ")
# print("Number of words in the string: ", len(WordList))
# for index, word in enumerate(WordList):
#     if word.isdigit():
#         print("Number in string: ", word)
#         print("Position of number in string: ", index)
# ==============================================================================================================

import os

import pandas as pd

path_Data = "./Stock Market"

AAPL = pd.read_csv(path_Data + "/AAPL.csv")
GOOG = pd.read_csv(path_Data + "/GOOG.csv")
MSFT = pd.read_csv(path_Data + "/MSFT.csv")

print(type(AAPL), type(GOOG), type(MSFT))
print("Shape AAPL : ", AAPL.shape, " Rows:", AAPL.shape[0], " Cols:", AAPL.shape[1])
print("Shape GOOG : ", GOOG.shape, " Rows:", GOOG.shape[0], " Cols:", GOOG.shape[1])
print("Shape MSFT : ", MSFT.shape, " Rows:", MSFT.shape[0], " Cols:", MSFT.shape[1])

display(AAPL.head())
display(GOOG.tail(10))
display(MSFT.head(3))
display(MSFT.iloc[2:4])
display(MSFT.loc[2:4])
