import csv
import sys
import math
import matplotlib.pyplot as plt
import numpy as np

def Q1(year, ice):
    plt.figure(figsize=(8,6))
    plt.plot(year, ice, marker='', linestyle='-', color='b')
    plt.xlabel("Year")
    plt.ylabel("NUmber of Frozen Days")

    plt.savefig("data_plot.jpg")

def Q2(year):
    year = np.array(year, dtype=np.float64)
    m = np.min(year)
    M = np.max(year)
    year_norm = (year-m)/(M-m)

    x_norm = np.column_stack((year_norm, np.ones_like(year_norm)))

    return x_norm

def Q3(year, ice):
    X=Q2(year)
    Y=np.array(ice, dtype=np.float64)

    wb = np.linalg.inv(X.T @ X) @ X.T @ Y

    return wb

def Q4(year, ice, learn, iterations):
    X=Q2(year)
    Y=np.array(ice, dtype=np.float64)

    weights=np.array([0.0, 0.0])
    n=len(Y)
    losses=[]

    print("Q4a:")
    for t in range(iterations):
        if t%10==0:
            print(weights)
        Y_p = np.dot(X, weights)
        err = Y_p - Y
        grad = (1/n) * np.dot(X.T, err)
        weights = weights - (learn*grad)
        loss = (1/(2*n))*np.sum(err**2)
        losses.append(loss) 
       


    print("Q4b:", learn)
    print("Q4c:", iterations)
    print("Q4d: "
          "I started with a smaller learning rate and 500 iterations." 
          " I increased the learning rate until the threshold was met convincingly." 
          " I then slowly decreased the iterations to be as low as could be")


    plt.plot(range(iterations), losses, color='b')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.jpg")  

def Q5(year, wb):
    w, b = wb
    m = np.min(year)
    M = np.max(year)

    x_norm = (2023-m)/(M-m)
    y_hat = w*x_norm + b

    print("Q5: " + str(y_hat))

def Q6(wb):
    w = wb[0]
    symbol=""

    if w>0:
        symbol=">"
    elif w<0:
        symbol="<"
    else:
        symbol="="

    print("Q6a: " + symbol)
    print("Q6b: "
          "When w>0: the number of ice days per year increases over time."
          " When w<0: the number of ice days per year decreases over time."
          " When w=0: the number of ice days per year does not have a trend over time.")
    
def Q7(year, wb):
    w, b = wb
    m = min(year)
    M = max(year)

    x = m - (b * (M - m)) / w

    print("Q7a: " + str(x))
    print("Q7b: This is a compelling prediction to me. This prediciton assumes" 
          " that lake mendota will continue to freeze less and less at a relatively slow rate"
          " given the difference between the oldest and newst data points in ice_data.csv."
          " However, it is worth noting that this data started during the industrial revolution,"
          " and has no data before to consider how the ice would change without it."
          " Rapid increases in technology and global industrialization would make ice melt faster, but efforts to stop"
          " climate change would increase the number of ice days" )

def main():
    filename = sys.argv[1]
    learning_rate = float(sys.argv[2]) 
    iterations = int(sys.argv[3])

    year=[]
    ice=[]

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            year.append(int(row[0]))
            ice.append(int(row[1]))

    #Q1 //
    Q1(year, ice)

    #Q2 //
    x_norm = Q2(year)
    print("Q2:")
    print(x_norm)

    #Q3 //
    wb = Q3(year, ice)
    print("Q3:") 
    print(wb)

    #Q4 //
    Q4(year, ice, learning_rate, iterations)

    #Q5 // 
    Q5(year, Q3(year, ice))

    #Q6 // 
    Q6(Q3(year, ice))

    #Q7 
    Q7(year, Q3(year, ice))
   
if __name__ == "__main__":
    main()