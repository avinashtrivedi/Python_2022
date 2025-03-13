# read the file
with open('question2_data.csv') as fp:
    data = fp.readlines()

# clean the data to get list of lists in integer form
data = [[int(j) for j in i.strip().split(',')] for i in data]

# variable to count the number of team processed
count = 0

# iterate through all the teams
for x in data:
    try:
        print(60*'-')
        
        # print wins,losses,ties
        print(f'Team {x[0]}')
        print(f'{x[1]} wins {x[2]} losses {x[3]} ties')

        # compute the total number of game played
        total_game = sum(x[1:])
        print(f'Total number of games played is {total_game},{16-total_game} games still remaining')
        
        # winning average
        print(f'The winning average is {x[1]/total_game:.4f}')

        # compare wins, losses with ties
        if x[3] >= x[1]:
            print('Number of games tied is greater than or equal to number won')
        else:
            print('Number of games tied is not greater than or equal to number won')

        if x[3] > x[2]:
            print('Number of games tied is greater than number lost')
        else:
            print('Number of games tied is not greater than number lost')

        # wip
        wip = (x[1] + x[3] - 3*x[2]) if (x[1] + x[3] - 3*x[2])>0 else 0
        print(f'The wip total is {wip}')
        
        count = count + 1
    except:
        print(f'Unable to process remaining info for Team {x[0]}')
        
# Total number team processed
print(f'\nTotal number team processed: {count}')