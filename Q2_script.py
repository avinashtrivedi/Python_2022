import pandas as pd
df = pd.read_csv('election_data.csv')
total = len(df)
x = df['Candidate'].value_counts()

winner = x.index[0]

print("Election Results")
print(30*'-')
print(f"Total Votes: {total}")
print(30*'-')
for name in sorted(x.index):
    votes = x[name]
    print(f"{name}: {round(votes*100/len(df),3)}% ({votes})")
print(30*'-')
print(f'Winner: {winner}')
print(30*'-')

fp = open('Q2.txt','w')

fp.write("Election Results")
fp.write('\n')
fp.write(30*'-')
fp.write('\n')
fp.write(f"Total Votes: {total}")
fp.write('\n')
fp.write(30*'-')
fp.write('\n')
for name in sorted(x.index):
    votes = x[name]
    fp.write(f"{name}: {round(votes*100/len(df),3)}% ({votes})")
    fp.write('\n')
fp.write(30*'-')
fp.write('\n')
fp.write(f'Winner: {winner}')
fp.write('\n')
fp.write(30*'-')
fp.close()