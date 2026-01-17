import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X={chr(i): 0 for i in range(ord('A'), ord('Z') + 1)} # create dict following a:0, b:1, etc
    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        for l in f:
            for c in l:
                if c.isalpha(): #only count letters
                    char = c.upper() #ignore case
                    if char in X:
                        X[char]+=1 
    return X



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

def getF(y, X, e, s, p):
    ret = math.log(p)
    for i in range(26):
        l = chr(i+ord('A'))
        c = X[l]

        prob=e[i]
        if y=='s':
            prob=s[i]
        
        if c>0:
            ret += c*math.log(prob)
    return ret
            
def main():
    filename = sys.argv[1]     # interpret the filename from the argument
    pE = float(sys.argv[2])
    pS = float(sys.argv[3])
    X = shred(filename) # shred and sort for alphabetical order
    e, s = get_parameter_vectors() # get vectors

    #Q1 -

    #Q2 
    X1 = X['A']
    X1loge1 = X1*math.log(e[0])
    X1logs1 = X1*math.log(s[0])

    #Q3
    Fe= getF('e', X, e, s, pE)
    Fs= getF('s', X, e, s, pS)

    #Q4
    dif = Fs - Fe
    if dif>=100:
        P_e_X = 0
    elif dif<=-100:
        P_e_X = 1
    else:
        P_e_X = 1 / (1+math.exp(dif))

    print("Q1")
    for let in sorted(X.keys()):
        print(f"{let} {X[let]}")

    print("Q2")
    print(f"{X1loge1: .4f}")
    print(f"{X1logs1: .4f}")

    print("Q3")
    print(f"{Fe: .4f}")
    print(f"{Fs: .4f}")

    print("Q4")
    print(f"{P_e_X: .4f}")

if __name__ == "__main__":
    main()

