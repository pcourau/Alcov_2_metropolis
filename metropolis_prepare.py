import numpy as np
import csv

lbda = 10
n = 10000

X=np.random.poisson(lbda,n)

with open("data_toy.csv",mode='w') as data_toy_file:
    data_toy_writer = csv.writer(data_toy_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    data_toy_writer.writerow(X)
