import pandas as pd

# shift+tab
# .tab
df = pd.read_csv('budget_data.csv')

month_total = len(df['Date'])

total = sum(df['Profit/Losses'])

df['Next'] = df['Profit/Losses'].shift(-1)
df['Changes'] = df['Next'] - df['Profit/Losses']
df['Date'] = df['Date'].shift(-1)
df.dropna(inplace=True)
change_avg = round(df['Changes'].mean(),2)

x = df.loc[df['Changes'].argmax()]

g_date = x['Date']

g_change = x['Changes']

y = df.loc[df['Changes'].argmin()]

d_date = y['Date']

d_change = y['Changes']

print('Financial Analysis')
print(30*'-')
print('Total Months:',month_total)
print(f'Total: ${total}')
print(f'Average Change: ${change_avg}')
print(f'Greatest Increase in Profits: {g_date} (${g_change})')
print(f'Greatest Decrease in Profits: {d_date} (${d_change})')

fp = open('Q1.txt','w')
fp.write('Financial Analysis')
fp.write('\n')
fp.write(30*'-')
fp.write('\n')
fp.write(f'Total Months: {month_total}')
fp.write('\n')
fp.write(f'Total: ${total}')
fp.write('\n')
fp.write(f'Average Change: ${change_avg}')
fp.write('\n')
fp.write(f'Greatest Increase in Profits: {g_date} (${g_change})')
fp.write('\n')
fp.write(f'Greatest Decrease in Profits: {d_date} (${d_change})')
fp.close()
