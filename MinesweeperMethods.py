#Some imports
import numpy as np
import matplotlib.pyplot as plt
import random
import queue
import time
from IPython.display import clear_output

#Functions that will be very useful for creating/updating mazes

'''
Define the grid to be working with

**inputs:
dim = dimension size of the grid
n = number of mines

**returns:
board = the grid to be worked with
'''

def environment(dim, n):
    #start with a dim by dim zero array

    board = np.zeros((dim,dim))

    while n > 0:
        i = random.randint(0, dim - 1)
        j = random.randint(0, dim - 1)

        if board[i][j] == 9:
            pass
        else:
            board[i][j] = 9
            n -= 1

    for i in range(0, dim):
        for j in range(0, dim):
            if board[i][j] == 9:
                continue

            #check all the neighbors
            mines = 0
            rightValid = False
            leftValid = False
            upValid = False
            downValid = False
            if j - 1 >= 0:
                leftValid = True
            if j + 1 < len(board):
                rightValid = True
            if i + 1 < len(board):
                downValid = True
            if i - 1 >= 0:
                upValid = True

            #check left
            if leftValid == True:
                #check left adjacent
                if board[i][j-1] == 9:
                    #mine is here
                    mines += 1
                else:
                    #no mine is here
                    pass
                #check left & up
                if upValid == True:
                    if board[i-1][j-1] == 9:
                        #mine is here
                        mines += 1
                    else:
                        #no mine is here
                        pass
                #check left & down
                if downValid == True:
                    if board[i+1][j-1] == 9:
                        #mine is here
                        mines += 1
                    else:
                        #no mine is here
                        pass

            #check right
            if rightValid == True:
                #check right adjacent
                if board[i][j+1] == 9:
                    #mine is here
                    mines += 1
                else:
                    #no mine is here
                    pass
                #check right & up
                if upValid == True:
                    if board[i-1][j+1] == 9:
                        #mine is here
                        mines += 1
                    else:
                        #no mine is here
                        pass
                #check right & down
                if downValid == True:
                    if board[i+1][j+1] == 9:
                        #mine is here
                        mines += 1
                    else:
                        #no mine is here
                        pass

            #check up adjacent
            if upValid == True:
                if board[i-1][j] == 9:
                    #mine is here
                    mines += 1
                else:
                    #no mine is here
                    pass

            #check down adjacent
            if downValid == True:
                if board[i+1][j] == 9:
                    #mine is here
                    mines += 1
                else:
                    #no mine is here
                    pass

            board[i][j] = mines

    return board

'''
A Method to Check the Neighbors of a Cell

**inputs:
possible_moves = array of coordinates for the remaining moves
coord = tuple containing the coordinates

**returns:
neighbors = the list of neighbors for the given coordinate
'''

def checkNeighbors(possible_moves, coord):
    neighbors = []
    i = coord[0]
    j = coord[1]

    if (i+1, j) in possible_moves:
        neighbors.append((i+1, j))

    if (i-1, j) in possible_moves:
        neighbors.append((i-1, j))

    if (i, j+1) in possible_moves:
        neighbors.append((i, j+1))

    if (i, j-1) in possible_moves:
        neighbors.append((i, j-1))

    if (i+1, j+1) in possible_moves:
        neighbors.append((i+1, j+1))

    if (i-1, j-1) in possible_moves:
        neighbors.append((i-1, j-1))

    if (i+1, j-1) in possible_moves:
        neighbors.append((i+1, j-1))

    if (i-1, j+1) in possible_moves:
        neighbors.append((i-1, j+1))

    return neighbors

'''
A Method to Update the Agent Board

**inputs:
coord = tuple containing the coordinates
main_board = the main board
agent_board = the agent board

**returns:
agent_board = the grid to be worked with
coord = tuple containing the coordinates
clue = number of adjacent mines
'''

def updateBoard(coord, main_board, agent_board):
    i = coord[0]
    j = coord[1]
    agent_board[i][j] = main_board[i][j]
    clue = agent_board[i][j]
    return agent_board, coord, clue

'''
A Method to Check the Number of Uncovered Mines for a Cell

**inputs:
board = the agent board
coord = tuple containing the coordinates

**returns:
mines = the number of neighboring mines
'''

def checkMines(board, coord):
    #check all the neighbors
    mines = 0
    i = coord[0]
    j = coord[1]
    rightValid = False
    leftValid = False
    upValid = False
    downValid = False
    if j - 1 >= 0:
        leftValid = True
    if j + 1 < len(board):
         rightValid = True
    if i + 1 < len(board):
         downValid = True
    if i - 1 >= 0:
        upValid = True

    #check left
    if leftValid == True:
        #check left adjacent
        if int(board[i][j-1]) == 9 or board[i][j-1] == 0.5:
            #mine is here
            mines += 1
        else:
            #no mine is here
            pass
        #check left & up
        if upValid == True:
            if int(board[i-1][j-1]) == 9 or board[i-1][j-1] == 0.5:
                #mine is here
                mines += 1
            else:
                #no mine is here
                pass
        #check left & down
        if downValid == True:
            if int(board[i+1][j-1]) == 9 or board[i+1][j-1] == 0.5:
                #mine is here
                mines += 1
            else:
                #no mine is here
                pass

    #check right
    if rightValid == True:
        #check right adjacent
        if int(board[i][j+1]) == 9 or board[i][j+1] == 0.5:
            #mine is here
            mines += 1
        else:
            #no mine is here
            pass
        #check right & up
        if upValid == True:
            if int(board[i-1][j+1]) == 9 or board[i-1][j+1] == 0.5:
                 #mine is here
                mines += 1
            else:
                #no mine is here
                pass
        #check right & down
        if downValid == True:
            if int(board[i+1][j+1]) == 9 or board[i+1][j+1] == 0.5:
                #mine is here
                mines += 1
            else:
                #no mine is here
                pass

    #check up adjacent
    if upValid == True:
        if int(board[i-1][j]) == 9 or board[i-1][j] == 0.5:
            #mine is here
            mines += 1
        else:
            #no mine is here
            pass

    #check down adjacent
    if downValid == True:
        if int(board[i+1][j]) == 9 or board[i+1][j] == 0.5:
            #mine is here
            mines += 1
        else:
            #no mine is here
            pass

    return mines

'''
Initialize blank equation to be used for inference

**inputs:
dim = the dimension size

**returns:
equation = a list of dim**2 zeros
'''

def equation(dim):
    equation = []
    while len(equation) < dim*dim:
        equation.append(0)
    return equation

'''
Reduce any inactive unknowns in a system of equations

**inputs:
matrix = a 2D array to be reduced

**returns:
a = the original matrix
b = the indexes to keep track of
c = the reduced matrix
'''

def reduce(matrix):
    #use numpy for easy calculations
    a = np.array(matrix)

    #list of indecies we keep track off
    b = []
    #go through the system of equations
    for i in range(0, len(a[0])):
        #we only want columns that are non-zero
        if sum(list(map(abs,a[:,i])))==0:
            pass
        else:
            b.append(i)

    #the reduced matrix
    c = []

    #reconstruct matrix without the zero columns
    for i in range(0, len(a)):
        c.append(list(a[i][b]))

    return a,b,c

'''
Generate Possible Solutions for a Given Matrix

**inputs:
matrix = system of equations to generate possible solutions for

**returns:
solutions = a list of potential solutions that need to be checked
'''

def generateSolutions(matrix):
    solutions = []

    #exact solutions
    if len(matrix[0]) <= 18:
        #create every possible binary combination
        for i in range(0,2**len(matrix[0])):
            a = list(bin(i)[2:])
            a = list(map(int, a))
            while len(a) < len(matrix[0]):
                a.insert(0, 0)
            solutions.append(a)

    #approximate solutions
    else:
        #create only a random sample of the total binary combinations
        moves = np.linspace(0,2**len(matrix[0]) - 1,2**len(matrix[0]))
        np.random.shuffle(moves)
        i = 0
        while i < 2**18:
            a = list(bin(int(moves[i]))[2:])
            a = list(map(int, a))
            while len(a) < len(matrix[0]):
                a.insert(0, 0)
            solutions.append(a)
            i += 1

    return solutions

'''
Generate probabilites that an active known is a mine

**inputs:
solutions = a list of potential solutions that needs to be checked
matrix = the system of equations to be used for validation
matrix_solutions = a vector that our answers will be check against

**returns:
prob_list = A list of probabilities
'''

def getProbs(solutions, matrix, matrix_solutions):
    matrix=np.array(matrix)
    solution_list = []
    prob_list = []

    #if the list of potential solutions given is empty, pass
    if len(solutions) == 0:
        pass
    else:
        #go through and dot each possibility with our matrix
        for item in solutions:
            result = matrix.dot(item)
            #if it works, keep it
            if np.array_equal(result, matrix_solutions):
                solution_list.append(item)

        solution_list = np.array(solution_list)

        if len(solution_list) == 0:
            pass
        else:
            #calculate the probabilites by averaging the solutions
            #sum of column/len of column
            for i in range(0, len(solution_list[0])):
                prob_list.append(sum(solution_list[:,i])/len(solution_list[:,i]))

    return prob_list

'''
Finds the expected number of squares worked out if we simulate clicking somewhere

**inputs:
q = the probability the coordinate simulated is a mine
agent_board = the current state of the agent's board
coord = the coordinate of the simulated cell
dim = the dimension of the board
isSafe = indicator to whether we are simulating a safe cell or a mine cell

**returns:
numSquaresWorkedOut = the number of squares worked out
'''

def solveForSquareHelper(q, agent_board, coord, dim, isSafe):
    #copy agent_board into sample board
    sample = np.zeros((dim,dim))
    for i in range(0, dim):
        for j in range(0, dim):
            sample[i][j] = agent_board[i][j]

    #a list of the expected squares that can definitely be worked out in sample board
    definiteSquares = []

    #mark square as either temp mine (.5) or temp safe (10)
    if(isSafe == 0):
        sample[coord[0]][coord[1]] = .5
    else:
        sample[coord[0]][coord[1]] = 10

    #populate a list of all the possible moves we can make
    #populate knowledge base with known squares
    possible_moves = []
    KB = []
    for i in range(0, dim):
        for j in range(0, dim):
            if(sample[i][j] == 11):
                possible_moves.append((i, j))

    for i in range(0, dim):
        for j in range(0, dim):
            if(sample[i][j] != .5 and sample[i][j] != 9 and sample[i][j] != 11 and sample[i][j] != 10):
                KB.append([(i,j), sample[i][j], len(checkNeighbors(possible_moves, (i,j))), sample[i][j]])

    #populate a list of the neighbors of the coord
    possible_neighbors = checkNeighbors(possible_moves,coord)

    #go through KB to find definite moves to count
    i = 0
    while i < len(KB):
        updated = False
        for item2 in range(0, len(KB)):

            KB[item2][2] = len(checkNeighbors(possible_moves, KB[item2][0]))

            if KB[item2][1] + checkMines(sample, KB[item2][0]) == KB[item2][3]:
                pass
            else:
                KB[item2][1] = KB[item2][3] - checkMines(sample, KB[item2][0])
                updated = True

        #check if definite safe
        if KB[i][1] == 0:
            x=checkNeighbors(possible_moves, KB[i][0])
            definiteSquares += x
            KB.remove(KB[i])
            for item in x:
                sample[item[0]][item[1]] = 10
                possible_moves.remove(item)
            i = 0
            continue

        #check if definite mine
        elif KB[i][1] == KB[i][2]:
            x=checkNeighbors(possible_moves, KB[i][0])
            definiteSquares += x
            KB.remove(KB[i])
            for item in x:
                sample[item[0]][item[1]] = 0.5
                possible_moves.remove(item)
            i = 0
            continue

        i += 1

    #remove duplicates in the list of definite coords identifiable and count number of remaining coords
    definiteSquares = list(dict.fromkeys(definiteSquares))
    numSquaresWorkedOut = len(definiteSquares)

    return numSquaresWorkedOut

'''
Simulates a coordinate being both a mine and safe, and works out the total squares we can determine

**inputs:
q = the probability the coordinate simulated is a mine
agent_board = the current state of the agent's board
coord = the coordinate of the simulated cell

**returns:
numSquares = the number of total squares worked out
'''

def expectedSquares(q, coord, agent_board):
    dim = len(agent_board)

    #work out the 'R' value
    squaresWorkedOut_m = solveForSquareHelper(q, agent_board, coord, dim, 0)

    #work out the 'S' value
    squaresWorkedOut_s = solveForSquareHelper(q, agent_board, coord, dim, 1)

    #q*R+(1-q)*S
    numSquares = q*squaresWorkedOut_m + (1-q)*squaresWorkedOut_s
    return numSquares

'''
Generate both probabilites and conditional probabilities

**inputs:
solutions = a list of potential solutions that needs to be checked
matrix = the system of equations to be used for validation
matrix_solutions = a vector that our answers will be check against

**returns:
cost_list = A list of the conditional probabilities
prob_list = A list of the original probabilities
'''

def getImprovedProb(potential_sol, matrix, matrix_solutions):
    matrix=np.array(matrix)
    solution_list = []

    #dot product of potential solutions and matrix
    for item in potential_sol:
        result = matrix.dot(item)
        #if equal, valid solution
        if np.array_equal(result, matrix_solutions):
            solution_list.append(item)

    solution_list = np.array(solution_list)

    prob_list = []
    cost_list=[]

    if len(solution_list) == 0:
        pass
    else:
        for i in range(0, len(solution_list[0])):
            #calculate probability
            #sum of column/len of column
            prob_list.append(sum(solution_list[:,i])/len(solution_list[:,i]))

        #calculate improved cost
        for i in range(0,len(prob_list)):
            #prob that the simulated cell is a mine and safe
            probM=prob_list[i]
            probS=1-probM

            #solutions not including the cell we are simulating
            solutionM=[]
            solutionS=[]

            #Split up solutions into safe and mine, for given var i
            #i column, j row
            for j in range(0,len(solution_list)):
                if solution_list[j][i]== 0:
                    solutionS.append(list(solution_list[j][0:i])+list(solution_list[j][i+1:]))
                else:
                    solutionM.append(list(solution_list[j][0:i])+list(solution_list[j][i+1:]))

            #calculate conditional probabilities
            prob_listM=[]
            prob_listS=[]

            solutionS=np.array(solutionS)
            solutionM=np.array(solutionM)

            if len(solutionM)==0:
                pass
            else:
                for i in range(0, len(solutionM[0])):
                    prob_listM.append(sum(solutionM[:,i])/len(solutionM[:,i]))
            if len(solutionS)==0:
                pass
            else:
                for i in range(0, len(solutionS[0])):
                    prob_listS.append(sum(solutionS[:,i])/len(solutionS[:,i]))

            #calculate the overall conditional probability for each permutation
            cost=list(probM*np.array(prob_listM))+list(probS*np.array(prob_listS))
            cost_list.append(cost)

    return cost_list, prob_list
